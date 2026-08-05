import math

import torch
import wandb
from torchvision.utils import save_image

from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet, utils

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
    loss_type="L2",
    unet_dim=128,
    unet_mults=(1, 2, 2, 2),
    unet_channels=3,
    training_objective="pred_x0",
)

run = wandb.init(config=default_hypers, project="bath-thesis", entity="jd202")
config = run.config

model = Unet(dim=config.unet_dim, dim_mults=config.unet_mults, channels=config.unet_channels)

diffusion = GaussianDiffusion(
    model,
    image_size=config.image_size,
    timesteps=config.timesteps,  # number of steps
    sampling_timesteps=config.sampling_timesteps,
    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type=config.loss_type,  # L1 or L2
    training_objective=config.training_objective,
)

trainer = Trainer(
    diffusion,
    "/Users/jake/Desktop/scp/cifar",
    train_batch_size=config.batch_size,
    training_learning_rate=config.learning_rate,
    num_training_steps=config.training_timesteps,  # total training steps
    num_samples=config.number_of_samples,
    gradient_accumulate_every=config.gradient_accumulation_rate,  # gradient accumulation steps
    ema_update_every=config.ema_update_rate,
    ema_decay=config.ema_decay,  # exponential moving average decay
    amp=config.use_amp,  # turn on mixed precision
    fp16=config.use_fp16,
    save_and_sample_every=config.save_and_sample_rate,
    wandb_run=run,
)

trainer.load("./results/loadins", "56")
trainer.ema.ema_model.eval()
with torch.no_grad():
    milestone = 10 // 1
    batches = utils.num_to_groups(config.number_of_samples, config.batch_size)
    all_images_list = list(map(lambda n: trainer.ema.ema_model.sample(batch_size=n), batches))

all_images = torch.cat(all_images_list, dim=0)
save_image(all_images, f"results/samples/sample-{milestone}.png", nrow=int(math.sqrt(config.number_of_samples)))
run.finish()
