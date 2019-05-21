import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

db_file = r'data/cells/db.csv'
df = pd.read_csv(db_file, names=['response', 'layer'], dtype={'response': 'object', 'layer': 'str'})
print(df.)
