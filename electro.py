import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyts.image.gaf import GramianAngularField
from multiprocessing import Pool
from scipy.misc import imsave
import os

from eda import get_common_sample_frequency


db_file = '/media/wd/data/cells/db.p'
df = pd.read_pickle(db_file)
df['sampling_rate'] = df['sampling_rate'].astype('float')
df['layer'] = df['layer'].replace(['6a', '6b'], 6)
df['layer'] = df['layer'].replace('2/3', 2)
df['layer'] = df['layer'].astype('int')
SAMPLE_RATE = get_common_sample_frequency(df)
IMG_SIZE = 224


def activity_to_image(activity: np.array) -> np.ndarray:
    gaf = GramianAngularField(image_size=224, method='d')
    image = np.empty((IMG_SIZE, IMG_SIZE, 3))
    segment_length = int(len(activity)/3)
    for i in range(3):
        segment = activity[i*segment_length:(i+1)*segment_length]
        image[:, :, i] = gaf.transform(segment.reshape(1, -1)).squeeze()
    # image= gaf.transform(activity.reshape(1, -1)).squeeze()
    return image


def file_to_image(cell_file_name: str, output_dir: str, stim_file_name: str = '', resample: int = 1):
    activity = np.load(cell_file_name)
    if stim_file_name:
        stim = np.load(stim_file_name)
        activity = activity[stim]
    activity = activity[::resample]
    image = activity_to_image(activity)
    new_file_name = os.path.splitext(os.path.basename(cell_file_name))[0]
    image_save_location = '{}{}.png'.format(output_dir, new_file_name)
    imsave(image_save_location, image)


def convert_cell(cell: int, df: pd.DataFrame, out_dir:str):
    cell_file_name = df.loc[cell]['response_file']
    stim_file_name = df.loc[cell]['stimulation_given']
    current_sample_rate =  df.loc[cell]['sampling_rate']
    file_to_image(cell_file_name, out_dir, stim_file_name, resample=int(current_sample_rate/SAMPLE_RATE))


def move_to_class_folder(cell, sort_by, out_dir):
    class_name = df.loc[cell][sort_by]
    class_dir = out_dir + str(class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    os.rename('{}{}.png'.format(out_dir,cell), '{}{}/{}.png'.format(out_dir, class_name, cell ))


if __name__ == '__main__':
    out_dir = 'data/images/3dgadf/'
    # pool = Pool()
    cells = df.index
    for cell in cells:
        # convert_cell(cell, df, out_dir)
        move_to_class_folder(cell, 'dendrite_type', out_dir)
