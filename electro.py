import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyts.image.gaf import GramianAngularField


def activity_to_image(activity: np.array):
    gaf = GramianAngularField()
    transform = gaf.fit_transform(activity.reshape(1, -1)).squeeze()
    return transform


if __name__ == '__main__':
    db_csv = 'data/ce'
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