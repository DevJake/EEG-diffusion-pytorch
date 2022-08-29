from torch.utils.data import DataLoader

from denoising_diffusion_pytorch.utils import EEGTargetsDataset, cycle

dataset = EEGTargetsDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=0)
dataloader = cycle(dataloader)
data = next(dataloader)

print(data)
