import math
import os
import random
import shutil
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from ema_pytorch import EMA
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision import utils
from tqdm.auto import tqdm


def exists(x):
    """
    This method checks if the given parameter is not equal to None, i.e., if it has a value.

    :param x: The value to be checked for None-status.
    """
    return x is not None


class GenericDataset(Dataset):
    def __init__(
        self, folder, image_size: int, exts: list = None, augment_horizontal_flip=False, convert_image_to=None
    ):
        """
        This class loads images from a given directory and resizes them to be square.

        :param folder: The folder that contains the files for this dataset.
        :param int image_size: The dimensions for the given image. All images will be converted to a square with these dimensions.
        :param list exts: A list of file extensions that this class should load, such as jpg and png.
        :param augment_horizontal_flip: If a horizontal (left-to-right) flip of the image should be performed.
        :param convert_image_to: A given lambda function specifying how to convert the input images
        for this dataset. This is applied before any other manipulations, such as resizing or
        horizontal flipping.
        """
        super().__init__()
        if exts is None:
            exts = ["jpg", "jpeg", "png", "tiff"]
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f"{folder}").glob(f"**/*.{ext}")]

        lambda_convert_function = (
            partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
        )
        # nn.Identity simply returns the input.
        # So, if convert_image_to_fn is None,
        # lambda_convert_function will just return whatever is input to it

        self.transform = T.Compose(
            [
                T.Lambda(lambda_convert_function),
                # Execute some lambda of code to convert an image of the dataset by, such as greyscaling.
                T.Resize(image_size),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        """
        Return the number of images in this dataset.
        """
        return len(self.paths)

    def __getitem__(self, index):
        """
        Return the given image for the given index in this dataset.
        """
        path = self.paths[index]
        with Image.open(path) as image:
            return self.transform(image)


def find_and_move_unsorted(src_dir, ftype: str):
    if len(os.listdir(f"{src_dir}/unsorted")) <= 0:
        pass
    # for path in Path(f'{src_dir}/unsorted').rglob(f'*.{ftype}'):
    for path in Path(f"{src_dir}/unsorted").rglob("*.*"):
        name = str(path).lower().split("/")[-1]
        p = None
        p = "penguin" if "penguin" in name else p
        p = "guitar" if "guitar" in name else p
        p = "flower" if "flower" in name else p

        assert p is not None, f"Could not sort file at path: {path}"
        print("Moving", name, "to", src_dir, p)
        os.makedirs(f"{src_dir}/{p}", exist_ok=True)
        os.rename(path, f"{src_dir}/{p}/{name}")

    shutil.rmtree(f"{src_dir}/unsorted")
    os.mkdir(f"{src_dir}/unsorted")


class EEGTargetsDataset(Dataset):
    def __init__(
        self,
        eeg_directory="./datasets/eeg",
        targets_directory="./datasets/targets",
        labels: list = None,
        shuffle_eeg=True,
        shuffle_targets=True,
        augment_horizontal_flip=True,
        file_types=None,
        unsorted_eeg_policy=None,
        unsorted_target_policy=None,
        image_size=None,
    ):
        """
        This class loads a dataset of EEG and target image pairs, and then determines the correct class label for each one.

        A typical directory structure is as follows:
        datasets/
        ├── eeg
        │   ├── flower
        │   ├── guitar
        │   ├── penguin
        │   └── unsorted
        └── target
            ├── flower
            ├── guitar
            └── penguin
            └── unsorted

        It is expected that the directories at the bottom layer are named after the label they contain.
        It is expected that the names match between the `eeg` and `targets` directory.

        Any files and directories in the 'unsorted' directory will be searched recursively
        and moved to the appropriate directory. Their appropriate directory will be inferred
        by the file's name.


        :param eeg_directory: The directory containing EEG images for training. These will be loaded recursively.
        :param targets_directory: The directory containing target images for training. These will be loaded recursively.
        :param labels: The list of labels to be used for training. If left blank, this defaults to guitar, penguin and flower.
        :param shuffle_eeg: If the EEG training images should be sampled from randomly.
        :param shuffle_targets: If the target training images for each respective EEG image should be sampled from randomly.
        :param file_types: The list of file types to load.
        """
        if image_size is None:
            image_size = [32, 32]
        if unsorted_target_policy is None:
            unsorted_target_policy = ["move", "delete-src-dirs"]
        if unsorted_eeg_policy is None:
            unsorted_eeg_policy = ["move", "delete-src-dirs"]
        if labels is None:
            labels = ["penguin", "guitar", "flower"]

        if file_types is None:
            file_types = ["jpg", "png"]

        self.file_types = file_types
        self.labels = labels
        self.eeg_directory = eeg_directory
        self.targets_directory = targets_directory
        self.shuffle_eeg = shuffle_eeg
        self.shuffle_targets = shuffle_targets
        self.data = defaultdict(lambda: defaultdict(list))
        self.indices = {"eeg": {}, "target": {}}
        self.image_size = image_size

        # TODO load eeg recursively, load targets recursively, generate labels for each, group them together

        os.makedirs(eeg_directory, exist_ok=True)
        os.makedirs(targets_directory, exist_ok=True)

        for label in labels:
            assert os.path.exists(f"{eeg_directory}/{label}"), f"The EEG directory for `{label}` does not exist."
            assert os.path.exists(f"{targets_directory}/{label}"), (
                f"The targets directory for `{label}` does not exist."
            )
        #
        # for ftype in file_types:
        #     find_and_move_unsorted(eeg_directory, ftype)
        #     find_and_move_unsorted(targets_directory, ftype)

        for label in labels:
            d0 = os.listdir(f"{eeg_directory}/{label}")
            d1 = os.listdir(f"{targets_directory}/{label}")

            d0 = [d for d in d0 if not d.startswith(".")]
            d1 = [d for d in d1 if not d.startswith(".")]

            if self.shuffle_eeg:
                random.shuffle(d0)

            if self.shuffle_targets:
                random.shuffle(d1)

            self.data["eeg"][label] = d0
            self.data["targets"][label] = d1

            # print(f'Loaded {len(d0)} images for EEG/{label}')
            # print(f'Loaded {len(d1)} images for Targets/{label}')

        self.transformTarget = T.Compose(
            [
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.Resize(self.image_size),
                T.ToTensor(),
            ]
        )

        self.transformEEG = T.Compose([T.Resize(self.image_size), T.ToTensor()])

    def __getitem__(self, index):
        """
        Retrieve an item in the dataset at the given index. If shuffling is enabled, the given index is ignored.

        The values returned are in a tuple, and in the order of:
        1. EEG image for label `L`.
        2. Target image for label `L`.
        3. Label `L`.
        """
        label = random.choice(self.labels)
        eeg_sample = random.choice(self.data["eeg"][label])
        target_sample = random.choice(self.data["targets"][label])

        eeg_path = Path(self.eeg_directory) / label / eeg_sample
        target_path = Path(self.targets_directory) / label / target_sample
        with Image.open(eeg_path) as eeg_image:
            eeg_tensor = self.transformEEG(eeg_image.convert("L"))
        with Image.open(target_path) as target_image:
            if target_image.mode in ("RGBA", "LA") or (
                target_image.mode == "P" and "transparency" in target_image.info
            ):
                foreground = target_image.convert("RGBA")
                background = Image.new("RGBA", foreground.size, "white")
                target_image = Image.alpha_composite(background, foreground).convert("RGB")
            else:
                target_image = target_image.convert("RGB")
            target_tensor = self.transformTarget(target_image)

        return eeg_tensor.repeat(3, 1, 1), target_tensor, label

    def __len__(self):
        return sum(len(d[label]) for label in self.labels for d in self.data.values())


class Trainer:
    """Train, sample, and checkpoint a diffusion model with Accelerate."""

    def __init__(
        self,
        diffusion_model,
        training_images_dir="./datasets/eeg",
        *,
        training_targets_dir="./datasets/targets",
        training_dataset=None,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_horizontal_flip=True,
        training_learning_rate=1e-4,
        num_training_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder="./results",
        amp=False,
        fp16=False,
        split_batches=True,
        convert_image_to_ext=None,
        wandb_run=None,
        dataloader_num_workers=0,
    ):
        super().__init__()

        if amp and fp16:
            raise ValueError("amp and fp16 are aliases; enable only one")

        dataloader_config = DataLoaderConfiguration(split_batches=split_batches)
        self.accelerator = Accelerator(
            dataloader_config=dataloader_config,
            mixed_precision="fp16" if amp or fp16 else "no",
            gradient_accumulation_steps=gradient_accumulate_every,
            device_placement=True,
        )

        assert has_int_square_root(num_samples), "The number of samples must have an integer square root"

        self.diffusion_model = diffusion_model
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = num_training_steps
        self.image_size = diffusion_model.image_size
        self.wandb_run = wandb_run

        self.train_images_dataset = (
            training_dataset
            if training_dataset is not None
            else EEGTargetsDataset(
                eeg_directory=training_images_dir,
                targets_directory=training_targets_dir,
                augment_horizontal_flip=augment_horizontal_flip,
                image_size=[self.image_size, self.image_size],
            )
        )
        dataloader = DataLoader(
            self.train_images_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=self.accelerator.device.type == "cuda",
            num_workers=dataloader_num_workers,
        )

        self.optimiser = Adam(diffusion_model.parameters(), lr=training_learning_rate, betas=adam_betas)
        self.diffusion_model, self.optimiser, dataloader = self.accelerator.prepare(
            self.diffusion_model,
            self.optimiser,
            dataloader,
        )
        self.train_eeg_targets_dataloader = cycle(dataloader)
        self.ema = EMA(
            self.accelerator.unwrap_model(self.diffusion_model),
            beta=ema_decay,
            update_every=ema_update_every,
        )

        result_path = Path(results_folder)
        if wandb_run is not None:
            result_path /= f"{wandb_run.name}-{wandb_run.id}"
        self.results_folder = result_path
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True, parents=True)

        self.step = 0

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.diffusion_model),
            "opt": self.optimiser.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
        torch.save(data, self.results_folder / f"model-{milestone}.pt")

    def load(self, path, milestone):
        checkpoint_path = Path(path) / f"model-{milestone}.pt"
        data = torch.load(checkpoint_path, map_location=self.accelerator.device, weights_only=True)

        model = self.accelerator.unwrap_model(self.diffusion_model)
        model.load_state_dict(data["model"])
        self.step = data["step"]
        self.optimiser.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0
                self.optimiser.zero_grad()

                for _ in range(self.gradient_accumulate_every):
                    eeg_sample, target_sample, _ = next(self.train_eeg_targets_dataloader)

                    with accelerator.autocast():
                        loss = self.diffusion_model((eeg_sample, target_sample))
                        loss = loss / self.gradient_accumulate_every
                    total_loss += loss.detach().item()
                    accelerator.backward(loss)

                self.optimiser.step()
                accelerator.wait_for_everyone()

                if self.wandb_run is not None and accelerator.is_main_process:
                    self.wandb_run.log({"total_training_loss": total_loss, "training_timestep": self.step})
                pbar.set_description(f"loss: {total_loss:.4f}")

                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = [self.ema.ema_model.sample(batch_size=n) for n in batches]

                        all_images = torch.cat(all_images_list, dim=0)
                        utils.save_image(
                            all_images,
                            self.results_folder / f"sample-{milestone}.png",
                            nrow=int(math.sqrt(self.num_samples)),
                        )
                        self.save(milestone)

                self.step += 1
                pbar.update(1)

        accelerator.print("Training complete!")


def cycle(dataloader):
    while True:
        yield from dataloader


def has_int_square_root(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def compute_L2_norm(t):
    """
    Compute the L2 normalised value for some given value t.
    :param t: The value to compute the L2 norm against.
    """
    return F.normalize(t, dim=-1)


def normalise_to_negative_one_to_one(img):
    return img * 2 - 1


def unnormalise_to_zero_to_one(t):
    return (t + 1) * 0.5


def upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def downsample(input_channel_dims, output_channel_dims=None):
    """
    This method creates a downsampling convolutional layer of the U-Net architecture.

    :param input_channel_dims: The channel dimensions for the input to the layer.
    :param output_channel_dims: The channel dimensions for the output of the layer.
    """
    return nn.Conv2d(
        in_channels=input_channel_dims,
        out_channels=default(output_channel_dims, input_channel_dims),
        kernel_size=4,
        stride=2,
        padding=1,
    )


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine beta schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    :param s: The parameter that dictates the 'shift' of the beta values.
    A lower and higher value will cause beta to begin at a higher and lower value, respectively.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def default(val, func):
    """
    This method simply checks if the parameter val exists.
    If it does not, parameter func is either returned,
    or executed if it is itself a function.

    :param val: The value to be checked for its existence. See method exists for more.
    :param func: The value or function to be returned or executed (respectively) if val does not exist.
    """
    if exists(val):
        return val
    return func() if callable(func) else func
