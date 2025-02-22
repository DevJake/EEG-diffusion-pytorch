import glob
import json
import os
import uuid
import util.preprocessing.pipeline as pipe

if __name__ == '__main__':
    samples_dir = './samples'
    variations = None

    with open(f'./variations.json', 'r') as f:
        variations = json.load(f)

    for w, _ in variations['window_overlap']:
        assert len(pipe.get_output_dims_by_factors(w * 1024)) > 0, f'No factor pairs exist for a window width of {w}!'

    combinations = [
        [window_overlap, use_channels, high_pass, low_pass]
        for window_overlap in variations['window_overlap']
        for use_channels in variations['use_channels']
        for high_pass in variations['high_pass']
        for low_pass in variations['low_pass']
    ]

    print(f'Calculated {len(combinations)} total variants...')
    print(f'Estimated configuration count is {len(combinations) * len(list(glob.iglob(samples_dir + "/**.json")))}')

    for i, sample in enumerate(glob.iglob(f'{samples_dir}/*.json')):
        print('Operating on config #', i)

        os.makedirs(f'{samples_dir}/generated/config_{i}', exist_ok=True)

        conf = None
        with open(sample, 'r') as f:
            conf = json.load(f)

        for wo, use_channels, high_pass, low_pass in combinations:
            window_size, window_overlap = wo

            config = conf.copy()

            # config['META.CONFIG_NAME'] += f'_window_size={window_size}'
            # config['META.CONFIG_NAME'] += f'_window_overlap={window_overlap}'
            # config['META.CONFIG_NAME'] += f'_use_channels={use_channels}'
            # config['META.CONFIG_NAME'] += f'_highpass={high_pass}'
            # config['META.CONFIG_NAME'] += f'_lowpass={low_pass}'

            unique_id = str(uuid.uuid4())

            config['META.CONFIG_NAME'] += '--' + unique_id

            config['PREPROCESSING.HIGH_PASS_FILTER.FREQ'] = high_pass
            config['PREPROCESSING.LOW_PASS_FILTER.FREQ'] = low_pass
            config['RENDER.WINDOW_OVERLAP'] = window_overlap
            config['RENDER.WINDOW_WIDTH'] = window_size
            config['RENDER.DO_PER_CHANNEL'] = use_channels

            with open(f'{samples_dir}/generated/config_{i}/{config["META.CONFIG_NAME"]}.json', 'w') as f_o:
                json.dump(config, f_o, indent=4, sort_keys=True)
