#!/usr/bin/env python3

import os
import uuid

from PIL import Image
from tqdm import tqdm

from util.preprocessing import pipeline
import json


def preprocess(eeg_data_dir='./data/subjects', output_dir='./data/outputs/preprocessing/', montage_file_name='ANTNeuro_montage'):
    assert os.path.exists(f'{eeg_data_dir}/{montage_file_name}.json'), \
        'The specified montage file could not be found! ' \
        'Please check it is in the EEG data directory root.'
    # TODO Process a dictionary of metadata/hyperparameter variations

    hyperparameters = {
        'PER_CHANNEL': True,
        'PREPROCESSING.MONTAGE': True,
        'PREPROCESSING.REMOVE_DC': True,
        'PREPROCESSING.LOW_PASS_FILTER': True,
        'PREPROCESSING.HIGH_PASS_FILTER': True,
        'PREPROCESSING.USE_ICA': True
    }  # TODO add settings to hyperparameters, such as high and low pass amounts, not just toggles for stuff

    sub_sess_pairs = []  # subject, session

    for subject in range(20):
        for session in range(3):
            path = f'{eeg_data_dir}/Subject {subject}/Session {session}/sub{subject}_sess{session}'
            if not os.path.isfile(f'{path}.fdt'):
                continue
            sub_sess_pairs.append((subject, session))

    pbar_subjects = tqdm(len(sub_sess_pairs), desc='Subjects and Sessions')

    for subject, session in sub_sess_pairs:
        unique_id = str(uuid.uuid5()) # A unique ID for this run

        raw = pipeline.load_eeg(subject, session)

        raw = pipeline.apply_montage(raw, f'{eeg_data_dir}/{montage_file_name}.json')
        raw = pipeline.remove_DC(raw)
        raw = pipeline.apply_filter(raw, low_freq=0.1, high_freq=50)
        ica = pipeline.compute_ICA(raw)
        ica = pipeline.remove_EOG(raw, ica)
        # TODO assert that the use_ica hyperparameter is True
        # ica = pipeline.remove_ECG(raw, ica) # Sometimes works, sometimes does not, seems to be an issue with MNE
        raw = pipeline.apply_ICA_to_RAW(raw, ica)
        del ica  # It is no longer needed, so we delete it from memory

        _, _, epochs, _ = pipeline.generate_events(raw)
        path = f'{eeg_data_dir}/preprocessed/Subject {subject}/Session {session}/{unique_id}'
        os.makedirs(f'{path}/sub_{subject}_sess_{session}_preprocessed.fif', exist_ok=True)
        raw.save()

        with open(f'{path}/sub_{subject}_sess_{session}_hyperparams.fif', 'w') as f:
            json.dump(hyperparameters, f, sort_keys=True, indent=4)

        del raw

        A, B, C = ['imagined', 'perceived'], ['guitar', 'penguin', 'flower'], ['text', 'sound', 'pictorial']
        select_epochs = pipeline.select_specific_epochs(epochs, A, B, C)
        del select_epochs

        cropped_epochs = pipeline.crop_epochs(select_epochs)

        print('All preprocessing now complete, saving images!')

        pbar_epochs = tqdm(len(cropped_epochs), position=0, desc='Epoch progress')

        for i, p in enumerate(zip(cropped_epochs, cropped_epochs.event_id)):
            epoch, name = p
            images = pipeline.generate_eeg_dataset(
                epoch.squeeze())  # Remove outer dimension as this is just 1, so useless
            # pbar_channels = tqdm(images.shape[0], position=1, desc='Channel progress')

            for c, channel in enumerate(images):
                dir = f'{output_dir}/subject_{subject}/session_{session}/channel_{c}'
                os.makedirs(dir, exist_ok=True)
                # pbar_event = tqdm(channel.shape[0], position=2, desc='Event progress')
                for e, event in enumerate(channel):
                    im = Image.fromarray(event, 'L')
                    im.save(f'{dir}/epoch_{i}_channel_{c}_event_{e}_{name}.jpg')

                    # pbar_event.update(1)

                # pbar_channels.update(1)
                # pbar_event.close()

            pbar_epochs.update(1)
            # pbar_channels.close()

        # pbar_epochs.close()

        pbar_subjects.update(1)
        print(f'Completed preprocessing for subject {subject}, session {session}')

        del cropped_epochs


preprocess()
