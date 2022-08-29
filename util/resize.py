import glob
import os

from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

targets = [
    (32, 32),
    (48, 48),
    (48, 32),
    (32, 16),
    (64, 32),
    (64, 48),
    (64, 64),
    (96, 96),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (256, 128),
    (512, 128),
    (1024, 512),
    (1024, 256),
    (96, 48)
]
target_dirs = ['flower', 'guitar', 'penguin']

for w, h in targets:
    for d in target_dirs:

        os.makedirs(f'{d}-{w}x{h}', exist_ok=True)
        for f in glob.iglob(f'./{d}/*.*'):
            print(w, h, f)
            name = f.split('/')[-1]
            img = Image.open(f)
            img = ImageOps.fit(img, (w, h), Image.Resampling.LANCZOS)
            img.save(f'{d}-{w}x{h}/{name}')
