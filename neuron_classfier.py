from typing import List
from multiprocessing import Pool, Manager, Process
import pandas as pd
import numpy as np
import pickle
import os
from scipy.misc import imsave
import warnings
from h5py.h5py_warnings import H5pyDeprecationWarning

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor

from electro import activity_to_image

warnings.filterwarnings('ignore', category=H5pyDeprecationWarning)


class NeuronClassifierSetup:
    def __init__(self, species: str, save_path: str, sample_rate: float = 50000.0):
        self.species = getattr(CellTypesApi, species.upper())
        # root_path = '{}/{}'.format(save_path, species)
        root_path = save_path
        self._paths = {'root': save_path, 'time_series': os.path.join('data', species, 'data/time_series/'),
                       'images': os.path.join('data', species, 'data/images/'), 'results': 'results/'}
        if not os.path.exists(root_path):
            [os.makedirs(path, exist_ok=True) for path in self._paths.values()]
        self._sample_rate = sample_rate
        self._ctc = CellTypesCache(manifest_file=root_path + '/cells/manifest.json')
        self.database = pd.DataFrame()

    def parallel_create_data(self, seconds_to_use_each_segment: List[float]):
        if not isinstance(seconds_to_use_each_segment, list):
            seconds_to_use_each_segment = [seconds_to_use_each_segment]
        cells = self._ctc.get_cells(species=[self.species])
        pool = Pool()
        manager = Manager()
        d = manager.dict()
        # p = Pool(processes=8)
        for i, cell in enumerate(cells):
            for seg in seconds_to_use_each_segment:
                pool.apply_async(self.create_data, args=(d, cell, i, seg))
        pool.close()
        pool.join()
        # cells = [x for x in cells if x['id'] == 313861539]
        # d =  dict()
        self.create_data(d, cells[0], 1, 3)
        df = pd.DataFrame.from_dict(d.copy()).transpose()
        df['sampling_rate'] = df['sampling_rate'].astype('float')
        if self.species in ['Mus musculus', 'mouse']:
            df['layer'] = df['layer'].replace(['6a', '6b'], 6)
            df['layer'] = df['layer'].replace('2/3', 2)
        df['layer'] = df['layer'].astype('int')
        try:
            df = df.drop('490278904_39')  # Found to be bad samples
        except:
            pass
        df = df[df['dendrite_type'].isin(['spiny', 'aspiny'])]
        with open(self._paths['root'] + '/db.p', 'wb') as f:
            pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)
        self.database = df




    def create_data(self, cell_db, cell, ind, segment_length: float):

        # cells = [x for x in cells if x['id'] == 567901050]
        # for ind, cell in enumerate(cells):
            cell_id = cell['id']
            data_set = self._ctc.get_ephys_data(cell_id)
            sweeps = self._ctc.get_ephys_sweeps(cell_id)
            noise_sweep_number = [x['sweep_number'] for x in sweeps
                                  if x['stimulus_name'] in ['Noise 1', 'Noise 2']
                                  and x['num_spikes'] is not None
                                  and x['num_spikes'] > 15]
            if not noise_sweep_number or not cell['dendrite_type'] in ['spiny', 'aspiny']:
                return
            # else:
            #     noise_sweep_number = noise_sweep_number[0]
            try:  # Make sure ephys file is not corrupted
                sweep_data = data_set.get_sweep(noise_sweep_number[0])
            except:
                corrupted_filename = self._ctc.get_cache_path(None, 'EPHYS_DATA', cell_id)
                os.remove(corrupted_filename)
                data_set = self._ctc.get_ephys_data(cell_id)
            for sweep_num in [noise_sweep_number[0]]:
                print('Processing cell: {} sweep: {}. Cell {}'.format(cell_id, sweep_num, ind + 1))
                this_cell_id = '{}_{}'.format(cell_id, sweep_num)
                sweep_data = data_set.get_sweep(sweep_num)
                ephys_feats = self._get_ephys_features(sweep_data, segment_length)

                # raw_data_dir = os.path.join(self._paths['time_series'], str(segment_length))
                # os.makedirs(raw_data_dir, exist_ok=True)
                # raw_data_file = os.path.join(raw_data_dir, this_cell_id)
                #
                # relevant_signal = range(*sweep_data['index_range'])
                # stimulation_given = np.where(sweep_data['stimulus'][relevant_signal] > 0)[0]
                # resample = int(sweep_data['sampling_rate'] / self._sample_rate)
                # response = sweep_data['response'][relevant_signal][stimulation_given][::resample]
                # response = self._truncate_segment(response, segment_length)
                # response_img = activity_to_image(response)
                # image_dir = os.path.join(self._paths['images'], str(segment_length))
                # os.makedirs(image_dir, exist_ok=True)
                # image_save_file = os.path.join(image_dir, this_cell_id + '.png')
                # #
                # imsave(image_save_file, response_img)
                # np.save(raw_data_file, response)  # .astype('float16'))

                cell_db[this_cell_id + '_' + str(segment_length)] = {**{'layer': cell['structure_layer_name'],
                                            'dendrite_type': cell['dendrite_type'],
                                            'structure_area_abbrev': cell['structure_area_abbrev'],
                                            'sampling_rate': sweep_data['sampling_rate'],
                                            'segment_length': segment_length}, **ephys_feats}



    def _get_cell_cache(self, cell) -> None:
        cell_id = cell['id']
        self._ctc.get_ephys_data(cell_id)

    def mass_download(self):
        pool = Pool(100)
        cells = self._ctc.get_cells(species=[self.species])
        pool.map(self._get_cell_cache, cells)

    @staticmethod
    def _truncate_segment(signal, segment_length, sample_rate=50000):
        signal = signal.reshape((3, signal.size//3))
        signal = signal[:, 0:int(segment_length * sample_rate)]
        return signal.flatten()

    
    def _get_ephys_features(self, sweep_data, segment_length):
        index_range = sweep_data["index_range"]
        sampling_rate = sweep_data["sampling_rate"]  # in Hz
        i = sweep_data["stimulus"][0:index_range[1] + 1] * 1e12  # in pA
        v = sweep_data["response"][0:index_range[1] + 1] * 1e3  # in mV
        stim_ind = np.arange(0, i.size)[(i > 0) & (np.arange(0, i.size) > 10000)]
        stim_ind_truncated = self._truncate_segment(stim_ind, segment_length, sampling_rate)
        drop_ind = np.setdiff1d(stim_ind, stim_ind_truncated)
        relevant_ind = np.ones(i.shape, dtype=bool)
        relevant_ind[drop_ind] = False
        v_truncated = v[relevant_ind]
        i_truncated = i[relevant_ind]


        t = np.arange(0, len(v_truncated)) * (1.0 / sampling_rate)

        result = {}
        sweep_ext = EphysSweepFeatureExtractor(t=t, v=v_truncated, i=i_truncated)
        sweep_ext.process_spikes()
        for key in sweep_ext.spike_feature_keys():
            try:
                result['mean_' + key] = np.mean(sweep_ext.spike_feature(key))
            except TypeError:
                continue

        return result

if __name__ == '__main__':
    nc = NeuronClassifierSetup('human', 'data')
    nc.create_data()