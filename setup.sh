cd ~/
git clone https://github.com/DevJake/EEG-diffusion-pytorch.git diffusion
cd diffusion
uv python install 3.14.4
uv sync --frozen --extra tracking
uv run accelerate config
sudo apt install rclone
mkdir -p ~/.config/rclone
nano ~/.config/rclone/rclone.conf
# Add in your rclone config to connect to the repository storing all EEG and Targets data
mkdir -p datasets/eeg/unsorted datasets/eeg/flower datasets/eeg/penguin datasets/eeg/guitar
mkdir -p datasets/targets/unsorted datasets/targets/flower datasets/targets/penguin datasets/targets/guitar
cd ~/diffusion/datasets/eeg
#rclone copy gc:/bath-thesis-data/data/outputs/preprocessing . -P
rclone copy gc:/bath-thesis-data/data/subjects/preprocessed/combined . -P
find . -name "*.tar.gz" -exec tar -xf {} \; # this will take some time to run...
find . -name "*.tar.gz" -exec rm -v {} \;
#find ./ -type f -exec mv --backup=numbered {} ./ -v \;
cd ~/diffusion/datasets/targets/
rclone copy gc:/bath-thesis-data/data/classes/32x32.tar . -P
tar -xf 32x32.tar
rm 32x32.tar
mv 32x32/flower-32x32/* flower/ & mv 32x32/guitar-32x32/* guitar/ & mv 32x32/penguin-32x32/* penguin/
rm 32x32 -r
cd ../..
uv run accelerate launch model.py

# Given the enormous size of model save files, and their frequent saving,
# you can use the following command in a tmux session to have them be backed
# up the the Google Cloud bucket, or another provider of choice. Source directories are not deleted,
# so do not worry about the model crashing from not being able to save!
# while sleep 120; do rclone move ~/diffusion/results/ gc:bath-thesis-data/data/trained_models/ -P; done
