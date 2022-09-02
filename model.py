import torch

import wandb
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

torch.cuda.empty_cache()
wandb.login()

# wandb.config.learning_rate = 3e-4
# wandb.config.training_timesteps = 5000
# wandb.config.sampling_timesteps = 250
# wandb.config.image_size = 32
# wandb.config.number_of_samples = 25
# wandb.config.batch_size = 512
# wandb.config.use_amp = False
# wandb.config.use_fp16 = True
# wandb.config.gradient_accumulation_rate = 2
# wandb.config.ema_update_rate = 10
# wandb.config.ema_decay = 0.995
# wandb.config.adam_betas = (0.9, 0.99)
# wandb.config.save_and_sample_rate = 1000
# wandb.config.do_split_batches = False
# wandb.config.timesteps = 1000
# wandb.config.loss_type = 'L2'
# wandb.config.unet_dim = 16
# wandb.config.unet_mults = (1, 2, 4, 8)
# wandb.config.unet_channels = 3
# wandb.config.training_objective = 'pred_x0'

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
    timesteps=1000,
    loss_type='L2',
    unet_dim=16,
    unet_mults=(1, 2, 4, 8),
    unet_channels=3,
    training_objective='pred_x0'
)

wandb.init(config=default_hypers, project='bath-thesis', entity='jd202')

# with open('./sweep.yaml') as f:
#     sweep_config = yaml.load(f, Loader=SafeLoader)
#
# sweep_id = wandb.sweep(sweep_config, entity='jd202', project='bath-thesis')


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

trainer.load('./results/loadins', '17')

wandb.watch(model)
wandb.watch(diffusion)

trainer.train()

wandb.finish()
