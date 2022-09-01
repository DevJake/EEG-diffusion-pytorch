labels = ['flower', 'penguin', 'guitar']
import os
from pathlib import Path

import tqdm

cpt = sum([len(files) for r, d, files in os.walk('./')])

t = 0
for path in tqdm.tqdm(Path('./').rglob(f'*.jpg'), initial=0, total=cpt):
    split = str(path).split('/')
    subject, session, name = split[0], split[1], split[-1]

    label = None
    label = 'penguin' if 'penguin' in name else label
    label = 'guitar' if 'guitar' in name else label
    label = 'flower' if 'flower' in name else label

    new_name = f'{label}/{str(t).zfill(8)}_{subject}_{session}_{name}'
    os.rename(path, f'./{new_name}')
    t += 1
