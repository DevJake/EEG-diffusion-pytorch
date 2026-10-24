<img src="./images/denoising-diffusion.png" width="500px"></img>

## Denoising Diffusion Probabilistic Model, in PyTorch

Implementation of <a href="https://arxiv.org/abs/2006.11239">Denoising Diffusion Probabilistic Models</a> in PyTorch. It is a new approach to generative modeling that may <a href="https://ajolicoeur.wordpress.com/the-new-contender-to-gans-score-matching-with-langevin-sampling/">have the potential</a> to rival GANs. It uses denoising score matching to estimate the gradient of the data distribution, followed by Langevin sampling to sample from the true distribution.

This implementation was transcribed from the [original diffusion implementation](https://github.com/hojonathanho/diffusion).

Youtube AI Educators - <a href="https://www.youtube.com/watch?v=W-O7AZNzbzQ">Yannic Kilcher</a> | <a href="https://www.youtube.com/watch?v=344w5h24-h8">AI Coffeebreak with Letitia</a> | <a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">Outlier</a>

<a href="https://huggingface.co/blog/annotated-diffusion">Annotated code</a> by Research Scientists / Engineers from <a href="https://huggingface.co/">🤗 Huggingface</a>

Update: Turns out none of the technicalities really matters at all | <a href="https://arxiv.org/abs/2208.09392">"Cold Diffusion" paper</a>

<img src="./images/sample.png" width="500px"><img>

[![PyPI version](https://badge.fury.io/py/denoising-diffusion-pytorch.svg)](https://badge.fury.io/py/denoising-diffusion-pytorch)

## Installation

Python 3.14 is required. To install the published package into an existing uv project:

```bash
uv add denoising-diffusion-pytorch
```

For development, clone the repository and reproduce the locked environment:

```bash
uv python install 3.14.6
uv sync --frozen --group dev
```

Repository tools have separate extras. Add `--extra tracking` for W&B scripts, `--extra preprocessing` for EEG preprocessing, or `--all-extras` for both.

## Usage

```python
import torch
from denoising_diffusion_pytorch import GaussianDiffusion, Unet

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,
    loss_type="l1",
)

eeg_images = torch.rand(8, 3, 128, 128)
target_images = torch.rand(8, 3, 128, 128)
loss = diffusion((eeg_images, target_images))
loss.backward()

sampled_images = diffusion.sample(batch_size=4)
assert sampled_images.shape == (4, 3, 128, 128)
```

The `Trainer` consumes matching EEG and target directory trees. Each root must contain the same class subdirectories.

```python
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet

model = Unet(dim=64, dim_mults=(1, 2, 4, 8))
diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,
    sampling_timesteps=250,
    loss_type="l1",
)

trainer = Trainer(
    diffusion,
    training_images_dir="datasets/eeg",
    training_targets_dir="datasets/targets",
    train_batch_size=32,
    training_learning_rate=8e-5,
    num_training_steps=700000,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    amp=True,
)
trainer.train()
```

Samples and model checkpoints are written to `./results` periodically.

## W&B tracking

Install the tracking group, initialise a run, and pass that run to `Trainer(wandb_run=run)`. The core package does not import W&B, so local training works without a W&B account.

## Multi-GPU training

The `Trainer` uses [Accelerate](https://huggingface.co/docs/accelerate/accelerator). Configure and launch it from the project root:

```bash
uv run accelerate config
uv run accelerate launch model.py
```

## Citations

```bibtex
@inproceedings{NEURIPS2020_4c5bcfec,
    author      = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
    pages       = {6840--6851},
    publisher   = {Curran Associates, Inc.},
    title       = {Denoising Diffusion Probabilistic Models},
    url         = {https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf},
    volume      = {33},
    year        = {2020}
}
```

```bibtex
@InProceedings{pmlr-v139-nichol21a,
    title       = {Improved Denoising Diffusion Probabilistic Models},
    author      = {Nichol, Alexander Quinn and Dhariwal, Prafulla},
    booktitle   = {Proceedings of the 38th International Conference on Machine Learning},
    pages       = {8162--8171},
    year        = {2021},
    editor      = {Meila, Marina and Zhang, Tong},
    volume      = {139},
    series      = {Proceedings of Machine Learning Research},
    month       = {18--24 Jul},
    publisher   = {PMLR},
    pdf         = {http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf},
    url         = {https://proceedings.mlr.press/v139/nichol21a.html},
}
```

```bibtex
@inproceedings{kingma2021on,
    title       = {On Density Estimation with Diffusion Models},
    author      = {Diederik P Kingma and Tim Salimans and Ben Poole and Jonathan Ho},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
    year        = {2021},
    url         = {https://openreview.net/forum?id=2LdBqxc1Yv}
}
```

```bibtex
@article{Choi2022PerceptionPT,
    title   = {Perception Prioritized Training of Diffusion Models},
    author  = {Jooyoung Choi and Jungbeom Lee and Chaehun Shin and Sungwon Kim and Hyunwoo J. Kim and Sung-Hoon Yoon},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2204.00227}
}
```

```bibtex
@article{Karras2022ElucidatingTD,
    title   = {Elucidating the Design Space of Diffusion-Based Generative Models},
    author  = {Tero Karras and Miika Aittala and Timo Aila and Samuli Laine},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2206.00364}
}
```

```bibtex
@article{Song2021DenoisingDI,
    title   = {Denoising Diffusion Implicit Models},
    author  = {Jiaming Song and Chenlin Meng and Stefano Ermon},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2010.02502}
}
```

```bibtex
@misc{chen2022analog,
    title   = {Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning},
    author  = {Ting Chen and Ruixiang Zhang and Geoffrey Hinton},
    year    = {2022},
    eprint  = {2208.04202},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@article{Qiao2019WeightS,
    title   = {Weight Standardization},
    author  = {Siyuan Qiao and Huiyu Wang and Chenxi Liu and Wei Shen and Alan Loddon Yuille},
    journal = {ArXiv},
    year    = {2019},
    volume  = {abs/1903.10520}
}
```
