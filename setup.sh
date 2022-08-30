cd ~/
#python3 -m pip install virtualenv # It is just simpler this way, trust me...
git clone https://github.com/DevJake/EEG-diffusion-pytorch.git diffusion
cd diffusion
#virtualenv venv
#source venv/bin/activate
sudo python3 setup.py install
#pip3 install wandb accelerate einops tqdm ema_pytorch torchvision
#pip3 install -r requirements.txt
#python3 -m wandb login
accelerate config
sudo apt install rclone
mkdir -p ~/.config/rclone
nano ~/.config/rclone/rclone.conf
# Add in your rclone config to connect to the repository storing all EEG and Targets data
mkdir -p datasets/eeg/unsorted datasets/eeg/flower datasets/eeg/penguin datasets/eeg/guitar
mkdir -p datasets/targets/unsorted datasets/targets/flower datasets/targets/penguin datasets/targets/guitar
cd ~/diffusion/datasets/eeg/unsorted
rclone copy gc:/bath-thesis-data/data/outputs/preprocessing . -P
find . -name "*.tar.gz" -exec tar -xf {} \; # this will take some time to run...
cd ~/diffusion/datasets/targets/
rclone copy gc:/bath-thesis-data/data/classes/32x32.tar . -P
tar -xf 32x32.tar
rm 32x32.tar
mv 32x32/flower-32x32/* flower/ & mv 32x32/guitar-32x32/* guitar/ & mv 32x32/penguin-32x32/* penguin/
rm 32x32 -r
cd ../..
accelerate launch model.py

#curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
#sudo python3 pytorch-xla-env-setup.py --version 1.12