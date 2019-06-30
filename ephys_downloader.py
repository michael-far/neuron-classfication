#!/usr/bin/env python -W ignore::DeprecationWarning
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor

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
esf = EphysSweepFeatureExtractor
OUT_DIR = 'data/time_series/'


def get_cell_cache(cell):
    cell_id = cell['id']
    data_set = ctc.get_ephys_data(cell_id)


def mass_download():
    pool = Pool(100)
    cells = ctc.get_cells(species=[CellTypesApi.MOUSE])
    pool.map(get_cell_cache, cells)


def get_ephys_features(sweep_data):
    index_range = sweep_data["index_range"]
    i = sweep_data["stimulus"][0:index_range[1] + 1]  # in A
    v = sweep_data["response"][0:index_range[1] + 1]  # in V
    i *= 1e12  # to pA
    v *= 1e3  # to mV

    sampling_rate = sweep_data["sampling_rate"]  # in Hz
    t = np.arange(0, len(v)) * (1.0 / sampling_rate)

    result = {}
    sweep_ext = EphysSweepFeatureExtractor(t=t, v=v, i=i)
    sweep_ext.process_spikes()
    for key in sweep_ext.spike_feature_keys():
        try:
            result['mean_' + key] = np.mean(sweep_ext.spike_feature(key))
        except TypeError:
            continue

    return result


def create_data(data_path):
    cell_db = {}
    for ind, cell in enumerate(cells):
        cell_id = cell['id']
        data_set = ctc.get_ephys_data(cell_id)
        sweeps = ctc.get_ephys_sweeps(cell_id)
        noise_sweep_number = [x['sweep_number'] for x in sweeps
                              if x['stimulus_name'] in ['Noise 1', 'Noise 2']
                              and x['num_spikes'] is not None
                              and x['num_spikes'] > 15]
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
        for sweep_num in [noise_sweep_number[0]]:
            print('Proccesing cell: {} sweep: {}. Cell {}/{}'.format(cell_id, sweep_num, ind+1, len(cells)))
            this_cell_id = '{}_{}'.format(cell_id, sweep_num)
            sweep_data = data_set.get_sweep(sweep_num)
            ephys_feats = get_ephys_features(sweep_data)
            raw_data_file = '{}/{}.npy'.format(OUT_DIR, this_cell_id)
            # raw_data_stim = '{}/ephys/raw_data/{}_stim.npy'.format(data_path, this_cell_id)
            # plt.plot(sweep_data['response'])
            # plt.show()
            relevant_signal = range(*sweep_data['index_range'])
            stimulation_given = np.where(sweep_data['stimulus'][relevant_signal] > 0)[0]
            resample = int(sweep_data['sampling_rate']/SAMPLE_RATE)
            response = sweep_data['response'][relevant_signal][stimulation_given][::resample]
            response_img = activity_to_image(response)
            image_save_location = '{}{}.png'.format('data/images/3dgadf/', this_cell_id)
            #
            imsave(image_save_location, response_img)
            np.save(raw_data_file, response)#.astype('float16'))
            # np.save(raw_data_file, response.astype('float16'))
            # np.save(raw_data_stim, stimulation_given)
            cell_db[this_cell_id] = { **{'layer': cell['structure_layer_name'],
                                     'dendrite_type': cell['dendrite_type'],
                                     'structure_area_abbrev': cell['structure_area_abbrev'],
                                     'sampling_rate': sweep_data['sampling_rate']}, **ephys_feats}
            # 'stimulation_given': raw_data_stim}

    df = pd.DataFrame(data=cell_db).transpose()
    df['sampling_rate'] = df['sampling_rate'].astype('float')
    df['layer'] = df['layer'].replace(['6a', '6b'], 6)
    df['layer'] = df['layer'].replace('2/3', 2)
    df['layer'] = df['layer'].astype('int')
    df = df[df['dendrite_type'].isin(['spiny', 'aspiny'])]
    df['file_name'] = df.index
    with open(data_path + '/cells/db.p', 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    create_data('/media/wd/data')

    
"""
plt.subplot(2,1,1)
plt.plot(np.linspace(0,9,len(relevant_signal)), sweep_data['stimulus'][relevant_signal])
plt.xlabel('Time [sec]')
plt.ylabel('mV')
plt.title('Stimulation')

plt.subplot(2,1,2)
plt.plot(np.linspace(0,9,len(relevant_signal)), sweep_data['response'][relevant_signal])
plt.xlabel('Time [sec]')
plt.ylabel('mV')
plt.title('Recording')
plt.subplots_adjust(hspace=0.5)

"""
