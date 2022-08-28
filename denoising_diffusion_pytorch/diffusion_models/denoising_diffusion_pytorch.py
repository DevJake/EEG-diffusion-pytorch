from collections import namedtuple
from random import random

import torch
import torch.nn.functional as F
from einops import reduce
from torch import nn
from tqdm.auto import tqdm

from denoising_diffusion_pytorch.utils import normalise_to_negative_one_to_one, \
    unnormalise_to_zero_to_one, extract, linear_beta_schedule, cosine_beta_schedule, default

# constants
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# TODO add documentation/descriptions to every method and class
# TODO add WandB.ai support
# TODO generate model diagram and per-layer parameter count


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            objective='pred_noise',  # TODO add new objective
            beta_schedule='cosine',
            p2_loss_weight_gamma=0.,
            # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1.
            # is recommended
            p2_loss_weight_k=1,
            ddim_sampling_eta=1.
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        # Why the negated assertion? Rewrite the assertion

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0'}, \
            'The given objective must be either pred_noise (predict noise) or pred_x0 ' \
            '(predict image start) '

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'Unknown beta schedule specified: {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        # The default number of sampling timesteps. Reduced for Improved DDPM

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # Helper function to convert function values from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # Log when the variance is clipped, as the posterior variance is zero
        # at the beginning of the diffusion chain (x_0 to x_T)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Calculate reweighting values for p2 loss
        register_buffer('p2_loss_weight',
                        (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_x0_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_xt(self, x_t, t, x0):
        """
        This method attempts to predict the gaussian noise for the forwards process, from x_T to x_0.
        :param x_t: The isotropic gaussian noise sample at the beginning of the forwards process.
        :param t: The number of sampling timesteps.
        """
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond=None):
        model_output = self.model(x, t, x_self_cond)
        predicted_noise, x_0 = None, None

        if self.objective == 'pred_noise':
            predicted_noise = model_output
            x_0 = self.predict_x0_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            predicted_noise = self.predict_noise_from_xt(x, t, model_output)
            x_0 = model_output  # The output of the model, x0

        return ModelPrediction(predicted_noise, x_0)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def compute_sample_for_timestep(self, x, t: int, x_self_cond=None, clip_denoised=True):
        """
        This method takes a single step in the sampling/forwards process. For example, in a forwards process with 250
        sampling steps, this method will be called 250 times.
        :param x: The current image for the given timestep t. If t=T, then we are at the beginning of the forwards
        process, and x will be an isotropic Gaussian noise sample.
        :param t: The current sampling timestep that defines x_T, where 0 <= t <= T.
        """
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond,
                                                                          clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.
        # Reset noise if t is zero, i.e., if we now have the output image of the model.
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, device):
        # batch_size = shape[0]

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.compute_sample_for_timestep(img, t, self_cond)

        img = unnormalise_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, device, clip_denoised=True):
        batch, \
        total_timesteps, sampling_timesteps, \
        eta, objective = shape[0], \
                         self.num_timesteps, self.sampling_timesteps, \
                         self.ddim_sampling_eta, self.objective
        # TODO slow sampling issue likely originates from here, not being put on the correct device

        times = torch.linspace(0., total_timesteps, steps=sampling_timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)
        # Begin image, xT, sampled as random noise
        # TODO need a way to specify a noise sample from the EEG forward process,
        #  not to randomly generate it

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='Sampling loop time step'):
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            self_cond = x_start if self.self_condition else None

            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond)

            if clip_denoised:
                x_start.clamp_(-1., 1.)

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = ((1 - alpha_next) - sigma ** 2).sqrt()

            noise = torch.randn_like(img) if time_next > 0 else 0.

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        img = unnormalise_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size, device):
        batch_size = 16 if batch_size is None else batch_size  # default value
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), device)

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.compute_sample_for_timestep(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        # TODO explore alternative loss types, unlikely they will be of value however
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise=None):
        # b, c, h, w = x_start.shape # Commented out as these values are not used

        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None  # TODO look into using this
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # Predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
            # TODO modify to substitute the target from x_start to a sample image
            #  from the target class. Also add a new objective type to support this.
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # TODO substitute x_start in for a different image. EEG to image, not EEG to EEG. 

        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalise_to_negative_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)

# Dataset classes


# Trainer class
