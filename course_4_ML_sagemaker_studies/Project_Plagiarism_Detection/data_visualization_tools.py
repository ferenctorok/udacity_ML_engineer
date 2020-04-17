import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def load_data(file):
    '''Loads a .csv file into X data and y label arrays.'''
    if os.path.exists(file):
        # reading the file into a dataframe:
        df = pd.read_csv(filepath_or_buffer=file, header=None, index_col=False)
        
        # creating the data and the label np.arrays:
        X = df.values[:, 1:]
        y = df.values[:, 0]
        
        return X, y
    else:
        print('file {} does not exist'.format(file))
        

def plot_projection(X, y):
    '''This functions creates a scatter plot for every projection of the data.'''
    
    n = X.shape[1]
    plt.figure(figsize=(10, 10))

    for i in range(n):
        for j in range(n):
            plot_index = i * n + j + 1
            plt.subplot(n, n, plot_index)
            plt.scatter(X[:, i], X[:, j], c=y)
    