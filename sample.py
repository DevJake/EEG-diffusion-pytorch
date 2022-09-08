import math

import torch
from torchvision.utils import save_image

import wandb
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, utils

torch.cuda.empty_cache()
wandb.login()

default_hypers = dict(
    learning_rate=3e-4,
    training_timesteps=1001,
    sampling_timesteps=250,
    image_size=32,
    number_of_samples=25,
    batch_size=256,
    use_amp=False,
    use_fp16=False,
    gradient_accumulation_rate=2,
    ema_update_rate=10,
    ema_decay=0.995,
    adam_betas=(0.9, 0.99),
    save_and_sample_rate=1000,
    do_split_batches=False,
    timesteps=4000,
    loss_type='L2',
    unet_dim=128,
    unet_mults=(1, 2, 2, 2),
    unet_channels=3,
    training_objective='pred_x0'
)

wandb.init(config=default_hypers, project='bath-thesis', entity='jd202')

model = Unet(
    dim=wandb.config.unet_dim,
    dim_mults=wandb.config.unet_mults,
    channels=wandb.config.unet_channels
)

diffusion = GaussianDiffusion(
    model,
    image_size=wandb.config.image_size,
    timesteps=wandb.config.timesteps,  # number of steps
    sampling_timesteps=wandb.config.sampling_timesteps,
    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type=wandb.config.loss_type,  # L1 or L2
    training_objective=wandb.config.training_objective
)

trainer = Trainer(
    diffusion,
    '/Users/jake/Desktop/scp/cifar',
    train_batch_size=wandb.config.batch_size,
    training_learning_rate=wandb.config.learning_rate,
    num_training_steps=wandb.config.training_timesteps,  # total training steps
    num_samples=wandb.config.number_of_samples,
    gradient_accumulate_every=wandb.config.gradient_accumulation_rate,  # gradient accumulation steps
    ema_update_every=wandb.config.ema_update_rate,
    ema_decay=wandb.config.ema_decay,  # exponential moving average decay
    amp=wandb.config.use_amp,  # turn on mixed precision
    fp16=wandb.config.use_fp16,
    save_and_sample_every=wandb.config.save_and_sample_rate
)

trainer.load('./results/loadins', '56')
trainer.ema.ema_model.eval()
with torch.no_grad():
    milestone = 10 // 1
    batches = utils.num_to_groups(wandb.config.number_of_samples, wandb.config.batch_size)
    all_images_list = list(map(lambda n: trainer.ema.ema_model.sample(batch_size=n), batches))

all_images = torch.cat(all_images_list, dim=0)
save_image(all_images, str(f'results/samples/sample-{milestone}.png'),
           nrow=int(math.sqrt(wandb.config.number_of_samples)))
