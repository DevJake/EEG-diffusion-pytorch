from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet
from denoising_diffusion_pytorch.utils import EEGTargetsDataset


class TinyDenoiser(nn.Module):
    channels = 3
    self_condition = False
    out_dim = 3

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, image, timestep, self_condition=None):
        del timestep, self_condition
        return self.projection(image)


class TinyPairedDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        eeg = torch.rand((3, 8, 8), generator=generator)
        target = torch.rand((3, 8, 8), generator=generator)
        return eeg, target, "test"


def make_diffusion(*, timesteps=4, sampling_timesteps=2, training_objective="pred_x0"):
    return GaussianDiffusion(
        TinyDenoiser(),
        image_size=8,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        training_objective=training_objective,
    )


def test_paired_dataset_loads_greyscale_and_transparent_images(tmp_path: Path):
    eeg_directory = tmp_path / "eeg" / "flower"
    targets_directory = tmp_path / "targets" / "flower"
    eeg_directory.mkdir(parents=True)
    targets_directory.mkdir(parents=True)
    Image.new("L", (8, 8), 128).save(eeg_directory / "eeg.png")
    Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(targets_directory / "target.png")
    dataset = EEGTargetsDataset(
        eeg_directory=eeg_directory.parent,
        targets_directory=targets_directory.parent,
        labels=["flower"],
        shuffle_eeg=False,
        shuffle_targets=False,
        augment_horizontal_flip=False,
        image_size=[8, 8],
    )

    eeg, target, label = dataset[0]

    assert eeg.shape == target.shape == (3, 8, 8)
    assert label == "flower"
    assert torch.allclose(target, torch.ones_like(target))


def test_unet_construction_and_forward_on_cpu():
    model = Unet(dim=8, dim_mults=(1, 2), channels=3, resnet_block_groups=4)
    output = model(torch.randn(2, 3, 8, 8), torch.tensor([0, 1]))

    assert output.shape == (2, 3, 8, 8)
    assert output.device.type == "cpu"


def test_diffusion_loss_noise_and_sampling_on_cpu():
    diffusion = make_diffusion()
    eeg = torch.rand(2, 3, 8, 8)
    target = torch.rand(2, 3, 8, 8)
    timestep = torch.tensor([0, 3])
    noise = torch.zeros_like(eeg)

    noised = diffusion.q_sample(eeg, timestep, noise=noise)
    loss = diffusion((eeg, target))
    loss.backward()
    samples = diffusion.sample(batch_size=2)

    assert noised.shape == eeg.shape
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert samples.shape == (2, 3, 8, 8)
    assert torch.all((0 <= samples) & (samples <= 1))


def test_pred_noise_objective_uses_the_supplied_noise():
    diffusion = make_diffusion(training_objective="pred_noise")
    with torch.no_grad():
        diffusion.learning_model.projection.weight.zero_()
        diffusion.learning_model.projection.bias.zero_()
    eeg = torch.zeros(2, 3, 8, 8)
    target = torch.zeros_like(eeg)
    noise = torch.ones_like(eeg)

    loss = diffusion.compute_loss_for_timestep(eeg, target, torch.tensor([0, 1]), noise=noise)

    assert torch.equal(loss, torch.tensor(1.0))


def test_standard_diffusion_sampling_on_cpu():
    diffusion = make_diffusion(timesteps=2, sampling_timesteps=None)

    samples = diffusion.sample(batch_size=1)

    assert samples.shape == (1, 3, 8, 8)
    assert samples.device.type == "cpu"


def test_trainer_runs_and_restores_checkpoint_on_cpu(tmp_path: Path):
    trainer = Trainer(
        make_diffusion(),
        training_dataset=TinyPairedDataset(),
        train_batch_size=2,
        num_training_steps=1,
        num_samples=1,
        results_folder=tmp_path,
        ema_update_every=1,
        save_and_sample_every=10,
        dataloader_num_workers=0,
    )

    trainer.train()
    trainer.save("smoke")
    checkpoint = tmp_path / "model-smoke.pt"
    expected = {
        name: value.detach().clone()
        for name, value in trainer.accelerator.unwrap_model(trainer.diffusion_model).state_dict().items()
    }

    with torch.no_grad():
        for parameter in trainer.diffusion_model.parameters():
            parameter.zero_()
    trainer.step = 99
    trainer.load(tmp_path, "smoke")

    restored = trainer.accelerator.unwrap_model(trainer.diffusion_model).state_dict()
    assert checkpoint.is_file()
    assert trainer.step == 1
    assert all(torch.equal(restored[name], value) for name, value in expected.items())
