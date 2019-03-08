"""
Create feature CSVs for train and test datasets
"""
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle
import base64

def featurization():
    # Load data-sets
    print("Loading data sets...")
    train_data = pd.read_csv('./data/train_data.csv', header=None, dtype=float).values
    test_data = pd.read_csv('./data/test_data.csv', header=None, dtype=float).values
    print("done.")

    # Create PCA object of the 15 most important components
    print("Creating PCA object...")
    pca = PCA(n_components=15, whiten=True)
    pca.fit(train_data[:, 1:])

    train_labels = train_data[:, 0].reshape([train_data.shape[0], 1])
    test_labels = test_data[:, 0].reshape([test_data.shape[0], 1])

    train_data = np.concatenate([train_labels, pca.transform(train_data[:, 1:])], axis=1)
    test_data = np.concatenate([test_labels, pca.transform(test_data[:, 1:])], axis=1)
    print("done.")

    # END NEW CODE

    print("Saving processed datasets and normalization parameters...")
    # Save normalized data-sets
    np.save('./data/processed_train_data', train_data)
    np.save('./data/processed_test_data', test_data)

    # Save learned PCA for future inference
    with open('./data/norm_params.json', 'w') as f:
        pca_as_string = base64.encodebytes(pickle.dumps(pca)).decode("utf-8")
        json.dump({ 'pca': pca_as_string }, f)

    print("done.")


if __name__ == '__main__':
    featurization()