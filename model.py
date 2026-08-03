import torch
import wandb

from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet

torch.cuda.empty_cache()
wandb.login()

# config.learning_rate = 3e-4
# config.training_timesteps = 5000
# config.sampling_timesteps = 250
# config.image_size = 32
# config.number_of_samples = 25
# config.batch_size = 512
# config.use_amp = False
# config.use_fp16 = True
# config.gradient_accumulation_rate = 2
# config.ema_update_rate = 10
# config.ema_decay = 0.995
# config.adam_betas = (0.9, 0.99)
# config.save_and_sample_rate = 1000
# config.do_split_batches = False
# config.timesteps = 1000
# config.loss_type = 'L2'
# config.unet_dim = 16
# config.unet_mults = (1, 2, 4, 8)
# config.unet_channels = 3
# config.training_objective = 'pred_x0'

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
    loss_type="L2",
    unet_dim=16,
    unet_mults=(1, 2, 4, 8),
    unet_channels=3,
    training_objective="pred_x0",
)

run = wandb.init(config=default_hypers, project="bath-thesis", entity="jd202")
config = run.config

# with open('./sweep.yaml') as f:
#     sweep_config = yaml.load(f, Loader=SafeLoader)
#
# sweep_id = wandb.sweep(sweep_config, entity='jd202', project='bath-thesis')


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

trainer.load("./results/loadins", "17")

run.watch(model)
run.watch(diffusion)

trainer.train()

run.finish()
