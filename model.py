from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size=128,
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
    num_training_steps=700000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=False  # turn on mixed precision
)

if __name__ == '__main__':
    # torch.multiprocessing.freeze_support()
    # freeze_support()
    trainer.train()
