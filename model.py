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
    train_batch_size=32,
    training_learning_rate=8e-5,
    num_training_steps=200000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=False,  # turn on mixed precision
    save_and_sample_every=100
)


def main():
    wandb.login()
    wandb.init(project='bath-thesis', entity='jd202')
    wandb.watch(model)
    wandb.watch(diffusion)
    # torch.multiprocessing.freeze_support()
    # freeze_support()
    trainer.train()
