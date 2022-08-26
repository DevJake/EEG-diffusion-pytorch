#!/usr/bin/env python3

import os

from PIL import Image
from tqdm import tqdm

from util.preprocessing import pipeline


def preprocess(output_dir='./outputs'):
    sub_sess_pairs = [  # subject, session
        (10, 1),
        (10, 2),
        (13, 1),
        (3, 1),
        (17, 1),
        (11, 1),
        (11, 2),
        (16, 1),
        (15, 1),
        (15, 2),
        (12, 1),
        (12, 2),
        (8, 2),
        (14, 2)
    ]

    pbar_subjects = tqdm(len(sub_sess_pairs), desc='Subjects and Sessions')

    for subject, session in sub_sess_pairs:

        raw = pipeline.load_eeg(subject, session)

        raw = pipeline.apply_montage(raw, './data/ANTNeuro_montage.json')
        raw = pipeline.remove_DC(raw)
        raw = pipeline.apply_filter(raw, low_freq=0.1, high_freq=50)
        ica = pipeline.compute_ICA(raw)
        ica = pipeline.remove_EOG(raw, ica)
        # ica = pipeline.remove_ECG(raw, ica) # Someties works, sometimes does not, seems to be an issue with MNE
        raw = pipeline.apply_ICA_to_RAW(raw, ica)
        del ica  # It is no longer needed, so we delete it from memory

        _, _, epochs, _ = pipeline.generate_events(raw)

        A, B, C = ['imagined', 'perceived'], ['guitar', 'penguin', 'flower'], ['text', 'sound', 'pictorial']
        select_epochs = pipeline.select_specific_epochs(epochs, A, B, C)
        cropped_epochs = pipeline.crop_epochs(select_epochs)

        print('All preprocessing now complete, saving images!')

        pbar_epochs = tqdm(len(cropped_epochs), position=0, desc='Epoch progress', leave=True)

        for i, p in enumerate(zip(cropped_epochs, cropped_epochs.event_id)):
            epoch, name = p
            images = pipeline.generate_eeg_dataset(
                epoch.squeeze())  # Remove outer dimension as this is just 1, so useless
            pbar_channels = tqdm(images.shape[0], position=1, desc='Channel progress', leave=True)

            for c, channel in enumerate(images):
                dir = f'{output_dir}/subject_{subject}/session_{session}/channel_{c}'
                os.makedirs(dir, exist_ok=True)
                pbar_event = tqdm(channel.shape[0], position=2, desc='Event progress', leave=True)
                for e, event in enumerate(channel):
                    im = Image.fromarray(event, 'L')
                    im.save(f'{dir}/epoch_{i}_channel_{c}_event_{e}_{name}.jpg')

                    pbar_event.update(1)

                pbar_channels.update(1)
                # pbar_event.close()

            pbar_epochs.update(1)
            # pbar_channels.close()

        # pbar_epochs.close()

        pbar_subjects.update(1)
        print(f'Completed preprocessing for subject {subject}, session {session}')

        del raw, ica, select_epochs, cropped_epochs


preprocess()
