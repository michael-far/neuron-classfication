from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from multiprocessing import Pool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import os


ctc = CellTypesCache(manifest_file='/media/wd/data/cells/manifest.json')
cells = ctc.get_cells(species=[CellTypesApi.MOUSE])


def get_cell_cache(cell):
    cell_id = cell['id']
    data_set = ctc.get_ephys_data(cell_id)

def mass_download():
    pool = Pool(100)
    cells = ctc.get_cells(species=[CellTypesApi.MOUSE])
    pool.map(get_cell_cache, cells)

def create_csv(data_path):
    cell_db = {}
    for cell in cells:
        cell_id = cell['id']
        data_set = ctc.get_ephys_data(cell_id)
        sweeps = ctc.get_ephys_sweeps(cell_id)
        noise_sweep_number = [x['sweep_number'] for x in sweeps if x['stimulus_name'] == 'Noise 1']
        if not noise_sweep_number:
            continue
        else:
            noise_sweep_number = noise_sweep_number[0]
        try:
            sweep_data = data_set.get_sweep(no
            ise_sweep_number)
        except:
            corrupted_filename = ctc.get_cache_path(None, 'EPHYS_DATA', cell_id)
            os.remove(corrupted_filename)
            data_set = ctc.get_ephys_data(cell_id)
            sweep_data = data_set.get_sweep(noise_sweep_number)

        raw_data_path = '{}/ephys/raw_data/{}.npy'.format(data_path, cell_id)
        # plt.plot(sweep_data['response'])
        # plt.show()
        np.save(raw_data_path, sweep_data['response'].astype('float16'))
        cell_db[cell_id] = {'response_file': raw_data_path,
                            'layer': cell['structure_layer_name'],
                            'dendrite_type': cell['dendrite_type'],
                            'structure_area_abbrev': cell['structure_area_abbrev']}

    df = pd.DataFrame(data=cell_db).transpose()
    with open(data_path + '/cells/db.p', 'wb') as f:
          pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    create_csv('/media/wd/data')
