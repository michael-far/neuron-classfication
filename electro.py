import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from pyts.image.gaf import GramianAngularField

def nwb2csv(nwb_file:str, csv_file:str):
    io = NWBHDF5IO(nwb_file, 'r')
    nwbfile_in = io.read()
    test_timeseries_in = nwbfile_in.acquisition['test_timeseries']
    print(test_timeseries_in.data[:])

def activity_to_image(activity: np.array):
    gaf = GramianAngularField()
    transform = gaf.fit_transform(activity.reshape(1, -1)).squeeze()
    return transform




if __name__ == '__main__':
    # file = 'data/electrophys/474626524_ephys.nwb'
    # nwb2csv(file, 'data/electrophys/474626524_ephys.csv')
    df = pd.read_csv('data/electrophys/activity_six4.csv')
    fig, axs = plt.subplots(3, 2)
    random_rows = np.random.randint(0, len(df), 3)
    for i, row_num in enumerate(random_rows):
        row = df.iloc[row_num]
        signal = np.asarray(row[:-2])
        img = activity_to_image(signal)
        axs[i][0].plot(signal)
        axs[i][0].set_title('Activity')
        axs[i][1].imshow(img)
        axs[i][1].set_title('Image transform')
    plt.show()