import wandb

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

wandb.login()
wandb.init(project='bath-thesis', entity='jd202')

wandb.learning_rate = 1e-4
wandb.training_timesteps = 400000
wandb.sampling_timesteps = 250
wandb.image_size = 32
wandb.number_of_samples = 25
wandb.batch_size = 256
wandb.use_amp = False
wandb.use_fp16 = False
wandb.gradient_accumulation_rate = 2
wandb.ema_update_rate = 10
wandb.ema_decay = 0.995
wandb.adam_betas = (0.9, 0.99)
wandb.save_and_sample_rate = 1000
wandb.do_split_batches = False
wandb.timesteps = 1000
wandb.loss_type = 'l1'
wandb.unet_dim = 64
wandb.unet_mults = (1, 2, 4, 8)
wandb.unet_channels = 3

model = Unet(
    dim=wandb.unet_dim,
    dim_mults=wandb.unet_mults,
    channels=wandb.unet_channels
)

diffusion = GaussianDiffusion(
    model,
    image_size=wandb.image_size,
    timesteps=wandb.timesteps,  # number of steps
    sampling_timesteps=wandb.sampling_timesteps,
    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type=wandb.loss_type  # L1 or L2
)

trainer = Trainer(
    diffusion,
    '/Users/jake/Desktop/scp/cifar',
    train_batch_size=wandb.batch_size,
    training_learning_rate=wandb.learning_rate,
    num_training_steps=wandb.training_timesteps,  # total training steps
    num_samples=wandb.number_of_samples,
    gradient_accumulate_every=wandb.gradient_accumulation_rate,  # gradient accumulation steps
    ema_update_every=wandb.ema_update_rate,
    ema_decay=wandb.ema_decay,  # exponential moving average decay
    amp=wandb.use_amp,  # turn on mixed precision
    fp16=wandb.use_fp16,
    save_and_sample_every=wandb.save_and_sample_rate
)

wandb.watch(model)
wandb.watch(diffusion)

trainer.train()
