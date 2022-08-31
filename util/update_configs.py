import wandb

api = wandb.Api()
run = api.run("jd202/bath-thesis/1fqllayw")

run.config['learning_rate'] = 3e-4
run.config['training_timesteps'] = 5000
run.config['sampling_timesteps'] = 250
run.config['image_size'] = 32
run.config['number_of_samples'] = 25
run.config['batch_size'] = 512
run.config['use_amp'] = False
run.config['use_fp16'] = True
run.config['gradient_accumulation_rate'] = 2
run.config['ema_update_rate'] = 10
run.config['ema_decay'] = 0.995
run.config['adam_betas'] = (0.9, 0.99)
run.config['save_and_sample_rate'] = 1000
run.config['do_split_batches'] = False
run.config['timesteps'] = 1000
run.config['loss_type'] = 'L2'
run.config['unet_dim'] = 16
run.config['unet_mults'] = (1, 2, 4, 8)
run.config['unet_channels'] = 3
run.config['training_objective'] = 'pred_x0'

run.update()
