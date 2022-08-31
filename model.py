import wandb

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=3
)

diffusion = GaussianDiffusion(
    model,
    image_size=32,
    timesteps=1000,  # number of steps
    sampling_timesteps=250,
    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type='l1'  # L1 or L2
)

trainer = Trainer(
    diffusion,
    '/Users/jake/Desktop/scp/cifar',
    train_batch_size=128,
    training_learning_rate=1e-4,
    num_training_steps=400000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=False,  # turn on mixed precision
    save_and_sample_every=1000
)

wandb.login()
wandb.init(project='bath-thesis', entity='jd202')
wandb.watch(model)
wandb.watch(diffusion)

# wandb.learning_rate = 1e-4
# wandb.training_timesteps = 400000
# wandb.sampling_timesteps = 250
# wandb.image_size = 32
# wandb.number_of_samples = 25
# wandb.batch_size = 256
# wandb.use_amp = False
# wandb.use_fp16 = False


trainer.train()
