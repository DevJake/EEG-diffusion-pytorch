import wandb

api = wandb.Api()
run = api.run("jd202/bath-thesis/1fqllayw")
run.config.update(
    {
        "learning_rate": 3e-4,
        "training_timesteps": 5000,
        "sampling_timesteps": 250,
        "image_size": 32,
        "number_of_samples": 25,
        "batch_size": 512,
        "use_amp": False,
        "use_fp16": True,
        "gradient_accumulation_rate": 2,
        "ema_update_rate": 10,
        "ema_decay": 0.995,
        "adam_betas": (0.9, 0.99),
        "save_and_sample_rate": 1000,
        "do_split_batches": False,
        "timesteps": 1000,
        "loss_type": "L2",
        "unet_dim": 16,
        "unet_mults": (1, 2, 4, 8),
        "unet_channels": 3,
        "training_objective": "pred_x0",
    },
    allow_val_change=True,
)
run.update()
