import preprocessing.eeg_to_dataset_pipeline

from tqdm import tqdm
from PIL import Image
import os

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

    for subject, session in sub_sess_pairs:

        raw = preprocessing.eeg_to_dataset_pipeline.load_eeg(subject, session)

        raw = preprocessing.eeg_to_dataset_pipeline.apply_montage(raw, './data/ANTNeuro_montage.json')
        raw = preprocessing.eeg_to_dataset_pipeline.remove_DC(raw)
        raw = preprocessing.eeg_to_dataset_pipeline.apply_filter(raw, low_freq=0.1, high_freq=50)
        ica = preprocessing.eeg_to_dataset_pipeline.compute_ICA(raw)
        ica = preprocessing.eeg_to_dataset_pipeline.remove_EOG(raw, ica)
        # ica = remove_ECG(raw, ica) # Someties works, sometimes does not, seems to be an issue with MNE
        raw = preprocessing.eeg_to_dataset_pipeline.apply_ICA_to_RAW(raw, ica)
        del ica  # It is no longer needed, so we delete it from memory

        events, event_ids, epochs, events_list = preprocessing.eeg_to_dataset_pipeline.generate_events(raw)

        A, B, C = ['imagined', 'perceived'], ['guitar', 'penguin', 'flower'], ['text', 'sound', 'pictorial']
        select_epochs = preprocessing.eeg_to_dataset_pipeline.select_specific_epochs(epochs, A, B, C)
        cropped_epochs = preprocessing.eeg_to_dataset_pipeline.crop_epochs(select_epochs)

        print('All preprocessing now complete, saving images!')

        from tqdm import tqdm
        from PIL import Image
        import os

        pbar_epochs = tqdm(len(cropped_epochs), position=0, desc='Epoch progress', leave=True)

        for i, p in enumerate(zip(cropped_epochs, cropped_epochs.event_id)):
            epoch, name = p
            images = preprocessing.eeg_to_dataset_pipeline.generate_eeg_dataset(epoch.squeeze())  # Remove outer dimension as this is just 1, so useless
            pbar_channels = tqdm(images.shape[0], position=1, desc='Channel progress', leave=True)

            for c, channel in enumerate(images):
                dir = f'{output_dir}/subject_{subject}/session_{session}/channel_{c}'
                os.makedirs(dir, exist_ok=True)
                pbar_event = tqdm(channel.shape[0], position=2, desc='Event progress', leave=True)
                for e, event in enumerate(channel):
                    im = Image.fromarray(event, 'L')
                    im.save(f'{dir}/epoch_{i}_channel_{c}_event_{k}_{name}.jpg')

                    pbar_event.update(1)

                pbar_channels.update(1)
                # pbar_event.close()

            pbar_epochs.update(1)
            # pbar_channels.close()

        # pbar_epochs.close()

        print(f'Completed preprocessing for subject {subject}, session {session}')
