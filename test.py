import pandas as pd
import numpy as np
import math

def normalize(data: pd.DataFrame):
    for col in data.columns:
        data[col] = data[col] / data[col].abs().max()

dict_data = {
    'age': [12, 13, 26, 42],
    'height': [48, 60, 66, 74],
    'weight': [120, 130, 140, 300]
}

data = pd.DataFrame.from_dict(dict_data)

print(data)
normalize(data)
print(data)