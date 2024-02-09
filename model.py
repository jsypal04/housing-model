import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers

def normalize(data: pd.DataFrame):
    for col in data.columns:
        data[col] = data[col] / data[col].abs().max()

def clean_cat_new_cols(data: pd.DataFrame):
    '''
    Converts all catagorical data to numerical data by adding a column for each catagorical entry
    '''
    cat_columns = data.columns[data.dtypes == 'object']
    # for each column in the catagorical columns Index
    for i in range(len(cat_columns)):
        # for each data entry in the column
        for j in range(len(data[cat_columns[i]])):
            # check to see if the entry has a coorisponding column
            entry = data[cat_columns[i]][j]
            if entry in data.columns:
                # if so make that entry a 1
                data.loc[j, entry] = 1
            # if there is no column with that header
            else:
                # create a new column and initialize it to all zeros
                loc = data.columns.get_loc(cat_columns[i]) + 1
                data.insert(loc, data[cat_columns[i]][j], np.zeros((len(data))))
                # set the current entry to 1
                data.loc[j, entry] = 1
        # delete the original catagorical column
        data.drop(cat_columns[i], axis=1, inplace=True)

def clean_cat_same_cols(data: pd.DataFrame):
    '''
    Converts catagorical data to numeric data by keeping the same number of columns
    Elimenates nan values in numeric data
    '''
    # STEP 1: create a dictionary mapping each column to a set of the possible data entries
    # initialize an empty dict mapping columns to a set of the possible data entries
    entries = dict()
    cat_columns = data.columns[data.dtypes == 'object']
    # for each column in the catagorical columns index
    for col in cat_columns:
        # add the column to the dict mapped to an empty set
        entries[col] = []
        # for each string data entry in the column
        for entry in data[col]:
            # add the string entry to the columns' set
            if not entry in entries[col]:
                entries[col].append(entry)

    # STEP 2: assign each string a number corrosponding to its index in the entries dictionary
    # for each column in the catagorical columns index
    for col in cat_columns:
        # for each entry in the column
        for id in range(len(data[col])):
            # reassign the entry to the index of the entry in the columns entries list
            data.loc[id, col] = entries[col].index(data.loc[id, col])

    # STEP 3: eliminate nan values in numeric data
    num_columns = data.columns[data.dtypes != 'object']
    # for each column in the numeric columns index
    for col in num_columns:
        # for each entry in the column
        for id in range(len(data[col])):
            # if the entry is nan replace it with 0
            if pd.isna(data[col][id]):
                data.loc[id, col] = 0

# read the testing data and labes
test_data = pd.read_csv('data/numerical-test2.csv')
test_data.pop(test_data.columns[0])

test_labels = pd.read_csv('data/sample_submission.csv')
test_labels.pop('Id')

# read the training data
data = pd.read_csv('data/numerical-train2.csv')
data.pop(data.columns[0]) # ged rid of extra id column
data.pop('Id')
labels = data.pop('SalePrice')

# normalize the data myself
normalize(data)
normalize(test_data)

# regression model
model = keras.Sequential([
    layers.Dense(128, input_dim=79, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='linear')
])
model.compile(
    loss='mean_squared_error', 
    optimizer='adam',
    metrics='binary_accuracy'
)
history = model.fit(
    data, 
    labels,
    epochs=20,
    verbose=1,
)

plt.plot(history.history['loss'])
plt.show()

print(model.evaluate(test_data, test_labels, verbose=1))
# predictions = model.predict(test_data)

# xpoints = np.array(range(len(test_labels)))
# plt.plot(xpoints, predictions, 'ro', markersize=2)
# plt.plot(xpoints, test_labels.to_numpy(), 'bo', markersize=2)

# plt.show()
