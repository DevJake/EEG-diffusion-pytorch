import math
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path

import torch
from PIL import Image
from accelerate import Accelerator
from ema_pytorch import EMA
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils
from tqdm.auto import tqdm


def exists(x):
    return x is not None


class Dataset(Dataset):
    def __init__(
            self,
            folder,  # TODO load images recursively.
            image_size: int,
            exts: list = None,
            augment_horizontal_flip=False,
            convert_image_to=None
    ):
        """
        This class loads images from a given directory and resizes them to be square.

        :param folder: The folder that contains the files for this dataset.
        :param int image_size: The dimensions for the given image. All images will be converted to a square with these dimensions.
        :param list exts: A list of extensions that this class should load.
        :param augment_horizontal_flip: If a horizontal (left-to-right) flip of the image should be performed.
        :param convert_image_to: A given lambda function specifying how to convert the input images
        for this dataset. This is applied before any other manipulations, such as resizing or
        horizontal flipping.
        """
        super().__init__()
        if exts is None:
            exts = ['jpg', 'jpeg', 'png', 'tiff']
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        lambda_convert_function = partial(convert_image_to, convert_image_to) \
            if exists(convert_image_to) \
            else nn.Identity()  # TODO determine what partial(...) does
        # nn.Identity simply returns the input.
        # So, if convert_image_to is None,
        # lambda_convert_function will just return whatever is input to it

        self.transform = T.Compose([
            T.Lambda(lambda_convert_function),
            # Execute some lambda of code to convert an image of the dataset by, such as greyscaling.
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        """
        Return the number of images to be loaded for this dataset.
        """
        return len(self.paths)

    def __getitem__(self, index):
        """
        Return the given image for the given index in this dataset.
        """
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


class Trainer(object):
    """
    This class is responsible for the training, sampling and saving loops that
    take place when interacting with the model.
    """

    def __init__(
            self,
            diffusion_model,
            training_images_dir,  # Folder where the training images exist
            # TODO add a secondary folder for the target image-class pairings
            # TODO add a means of loading and reading in those image-class pairings
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            augment_horizontal_flip=True,  # Flip image from left-to-right
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            num_samples=25,
            results_folder='./results',
            amp=False,  # Used mixed precision during training
            fp16=False,  # Use Floating-Point 16-bit precision
            # TODO might be able to enable fp16 without affecting amp,
            #  allowing for the model to train on TPUs
            split_batches=True,
            convert_image_to_ext=None  # A given extension to convert image types to
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_square_root(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        self.train_images_dataset = Dataset(training_images_dir, self.image_size,
                                            augment_horizontal_flip=augment_horizontal_flip,
                                            convert_image_to=convert_image_to_ext)
        dataloader = DataLoader(self.train_images_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                                num_workers=cpu_count())

        dataloader = self.accelerator.prepare(dataloader)
        self.train_images_dataloader = cycle(dataloader)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.train_images_dataloader).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        # TODO in goes our EEG data. Need to also pass in its class label (guitar/penguin/flower)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(
                                map(lambda n: self.ema.ema_model.sample(batch_size=n, device=device), batches))
                            # TODO update the above to support passing of the device
                            #  for the classes ContinuousTimeDiff. and ElucidatedDiff.
                            #  Also check other classes for potential uses/conflicts.

                        all_images = torch.cat(all_images_list, dim=0)
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'),
                                         nrow=int(math.sqrt(self.num_samples)))
                        self.save(milestone)
                        # TODO have this run for every self.save_and_sample_every,
                        #  but without sampling (currently broken on TPUs)

                self.step += 1
                pbar.update(1)

        accelerator.print('Training complete!')


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_square_root(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to(img_type, image):
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
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


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
    # TODO add comment on what parameter 's' does/is
    """
    Cosine beta schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
