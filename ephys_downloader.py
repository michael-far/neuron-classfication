#!/usr/bin/env python -W ignore::DeprecationWarning
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from multiprocessing import Pool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import os
from scipy.misc import imsave
import warnings
from h5py.h5py_warnings import H5pyDeprecationWarning
from electro import activity_to_image

warnings.filterwarnings('ignore', category=H5pyDeprecationWarning)

SAMPLE_RATE = 50000.0 # Hz, based on EDA
ctc = CellTypesCache(manifest_file='/media/wd/data/cells/manifest.json')
cells = ctc.get_cells(species=[CellTypesApi.MOUSE])
OUT_DIR = 'data/images/3dgadf/'


def get_cell_cache(cell):
    cell_id = cell['id']
    data_set = ctc.get_ephys_data(cell_id)


def mass_download():
    pool = Pool(100)
    cells = ctc.get_cells(species=[CellTypesApi.MOUSE])
    pool.map(get_cell_cache, cells)


def create_data(data_path):
    cell_db = {}
    for cell in cells:
        cell_id = cell['id']
        data_set = ctc.get_ephys_data(cell_id)
        sweeps = ctc.get_ephys_sweeps(cell_id)
        noise_sweep_number = [x['sweep_number'] for x in sweeps
                              if x['stimulus_name'] in ['Noise 1', 'Noise 2']
                              and x['num_spikes'] is not None
                              and x['num_spikes'] > 10]
        if not noise_sweep_number:
            continue
        # else:
        #     noise_sweep_number = noise_sweep_number[0]
        try:  # Make sure ephys file is not corrupted
            sweep_data = data_set.get_sweep(noise_sweep_number[0])
        except:
            corrupted_filename = ctc.get_cache_path(None, 'EPHYS_DATA', cell_id)
            os.remove(corrupted_filename)
            data_set = ctc.get_ephys_data(cell_id)
        for sweep_num in noise_sweep_number:
            print('Proccesing cell: {} sweep: {}'.format(cell_id, sweep_num))
            this_cell_id = '{}_{}'.format(cell_id, sweep_num)
            sweep_data = data_set.get_sweep(sweep_num)
            raw_data_file = '{}/ephys/raw_data/{}.npy'.format(data_path, this_cell_id)
            # raw_data_stim = '{}/ephys/raw_data/{}_stim.npy'.format(data_path, this_cell_id)
            # plt.plot(sweep_data['response'])
            # plt.show()
            relevant_signal = range(*sweep_data['index_range'])
            stimulation_given = np.where(sweep_data['stimulus'][relevant_signal] > 0)[0]
            resample = int(sweep_data['sampling_rate']/SAMPLE_RATE)
            response = sweep_data['response'][relevant_signal][stimulation_given][::resample]
            response_img = activity_to_image(response)
            image_save_location = '{}{}.png'.format(OUT_DIR, this_cell_id)

            imsave(image_save_location, response_img)
            # np.save(raw_data_file, sweep_data['response'][relevant_signal][stimulation_given])#.astype('float16'))
            # np.save(raw_data_file, sweep_data['response'][relevant_signal][stimulation_given].astype('float16'))
            # np.save(raw_data_stim, stimulation_given)
            cell_db[this_cell_id] = {'layer': cell['structure_layer_name'],
                                     'dendrite_type': cell['dendrite_type'],
                                     'structure_area_abbrev': cell['structure_area_abbrev'],
                                     'sampling_rate': sweep_data['sampling_rate']}
            # 'stimulation_given': raw_data_stim}

    df = pd.DataFrame(data=cell_db).transpose()
    df['sampling_rate'] = df['sampling_rate'].astype('float')
    df['layer'] = df['layer'].replace(['6a', '6b'], 6)
    df['layer'] = df['layer'].replace('2/3', 2)
    df['layer'] = df['layer'].astype('int')
    with open(data_path + '/cells/db.p', 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    create_data('/media/wd/data')
