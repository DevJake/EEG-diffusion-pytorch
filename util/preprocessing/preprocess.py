#!/usr/bin/env python3
import glob
import os
import uuid

from PIL import Image
from tqdm import tqdm

from util.preprocessing import pipeline
import json


def preprocess(eeg_data_dir='./data/subjects', output_dir='./data/outputs/preprocessing/',
               montage_file_name='ANTNeuro_montage', hypers=None):
    assert os.path.exists(f'{eeg_data_dir}/{montage_file_name}.json'), \
        'The specified montage file could not be found! ' \
        'Please check it is in the EEG data directory root.'
    # TODO Process a dictionary of metadata/hyperparameter variations
    # TODO Load hyperparameter configurations from a given directory, run each

    if hypers is None:
        # TODO load the default hyperparameters from a configuration file
        hypers = {
            'RENDER.DO_PER_CHANNEL': True,
            'PREPROCESSING.DO_MONTAGE': True,
            'PREPROCESSING.DO_REMOVE_DC': True,
            'PREPROCESSING.DO_LOW_PASS_FILTER': True,
            'PREPROCESSING.DO_HIGH_PASS_FILTER': True,
            'PREPROCESSING.DO_USE_ICA': True,
            'PREPROCESSING.DO_REMOVE_EOG': True,
            'PREPROCESSING.LOW_PASS_FILTER.FREQ': 0.1,
            'PREPROCESSING.HIGH_PASS_FILTER.FREQ': 50,
            'RENDER.WINDOW_WIDTH': 1,
            'RENDER.WINDOW_OVERLAP': 0.5,
            'META.CONFIG_NAME': 'config-1',
            'META.CONFIG_COMMENT': 'The default config for any EEG preprocessing.'
        }

    sub_sess_pairs = []  # subject, session

    for subject in range(20):
        for session in range(3):
            path = f'{eeg_data_dir}/Subject {subject}/Session {session}/sub{subject}_sess{session}'
            if not os.path.isfile(f'{path}.fdt'):
                continue
            sub_sess_pairs.append((subject, session))

    print(f'Now executing config \'{hypers["META.CONFIG_NAME"]}\' '
          f'against the following subject and session pairs: {sub_sess_pairs}')
    pbar_subjects = tqdm(len(sub_sess_pairs), desc='Subjects and Sessions')
    k = 0
    for subject, session in sub_sess_pairs:
        k += 1
        print(f'Now preprocessing data for Subject {subject}, session {session}. Progress = {k}/{len(sub_sess_pairs)}')
        try:
            unique_id = str(uuid.uuid4())  # A unique ID for this run
            print('The unique ID for this subject/session pairing is', unique_id)
            hypers['META.UUID'] = unique_id

            raw = pipeline.load_eeg(subject, session)
            # TODO account for hyperparameter value changes
            if hypers['PREPROCESSING.DO_MONTAGE']:
                raw = pipeline.apply_montage(raw, f'{eeg_data_dir}/{montage_file_name}.json')

            if hypers['PREPROCESSING.DO_REMOVE_DC']:
                raw = pipeline.remove_DC(raw)

            if hypers['PREPROCESSING.DO_LOW_PASS_FILTER']:
                raw = pipeline.apply_filter(raw, low_freq=hypers['PREPROCESSING.LOW_PASS_FILTER.FREQ'], high_freq=None)

            if hypers['PREPROCESSING.DO_HIGH_PASS_FILTER']:
                raw = pipeline.apply_filter(raw, low_freq=None, high_freq=hypers['PREPROCESSING.HIGH_PASS_FILTER.FREQ'])

            ica = None
            if hypers['PREPROCESSING.DO_USE_ICA']:
                ica = pipeline.compute_ICA(raw)

                if hypers['PREPROCESSING.DO_REMOVE_EOG']:
                    ica = pipeline.remove_EOG(raw, ica)
            # ica = pipeline.remove_ECG(raw, ica) # Sometimes works, sometimes does not, seems to be an issue with MNE
                raw = pipeline.apply_ICA_to_RAW(raw, ica)
            del ica  # It is no longer needed, so we delete it from memory

            _, _, epochs, _ = pipeline.generate_events(raw)
            path = f'{eeg_data_dir}/preprocessed/Subject {subject}/Session {session}/{hypers["META.CONFIG_NAME"]}/{unique_id} '
            os.makedirs(path, exist_ok=True)
            raw.save(f'{path}/sub_{subject}_sess_{session}_preprocessed_raw.fif')

            with open(f'{path}/sub_{subject}_sess_{session}_hyperparams.json', 'w') as f:
                json.dump(hypers, f, sort_keys=True, indent=4)

            del raw

            A, B, C = ['imagined', 'perceived'], ['guitar', 'penguin', 'flower'], ['text', 'sound', 'pictorial']
            select_epochs = pipeline.select_specific_epochs(epochs, A, B, C)

            cropped_epochs = pipeline.crop_epochs(select_epochs)
            del select_epochs

            print('All preprocessing now complete, saving images!')

            pbar_epochs = tqdm(len(cropped_epochs), desc='Epoch progress')

            for i, p in enumerate(zip(cropped_epochs, cropped_epochs.event_id)):
                epoch, name = p
                images = pipeline.generate_eeg_dataset(
                    epoch.squeeze(),
                    per_channel=hypers['RENDER.DO_PER_CHANNEL'],
                    window_width_seconds=hypers['RENDER.WINDOW_WIDTH'],
                    window_overlap_seconds=hypers['RENDER.WINDOW_OVERLAP']
                )  # Remove outer dimension as this is just 1, so useless
                # pbar_channels = tqdm(images.shape[0], position=1, desc='Channel progress')

                for c, channel in enumerate(images):
                    dir = f'{output_dir}/subject_{subject}/session_{session}/{hypers["META.CONFIG_NAME"]}/{unique_id}/channel_{c}'
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

                print(f'Finished saving images for epoch {i}. Name={name}')

            # pbar_epochs.close()

            pbar_subjects.update(1)
            print(f'Completed preprocessing for subject {subject}, session {session}')

            del cropped_epochs
        except Exception as e:
            print(f'Encountered an exception for Subject {subject} session {session}. Stack trace is as follows...')
            print(e)
            continue


def load_and_process_hyperparameters(dir: str):
    print(f'Loading configs from dir: {dir}')
    configs = []

    for config_path in glob.iglob(f'{dir}/**/*.json'):
        with open(config_path, 'r') as f:
            configs.append(json.load(f))
            print('Loaded config:', json.load(f))

    print(f'Found {len(configs)} to be processed...')
    return configs


if __name__ == '__main__':
    num_cpu = '8'
    os.environ['OMP_NUM_THREADS'] = num_cpu
    preprocess()
    ALLOW_DEFAULT_CONFIG = False

    configs = load_and_process_hyperparameters('./data/configurations')
    for i, config in configs:
        if config['META.CONFIG_NAME'] == 'default-config' and not ALLOW_DEFAULT_CONFIG:
            continue
        print(f'Now processing with the following configuration ({i} of {len(configs)}):')
        print(config)
        try:
            preprocess(hypers=config)
        except Exception as e:
            print('An exception has been thrown, could not perform preprocessing for the given config...')
            print(config)

    print('Every config has now been processed, hurray! Terminating...')
