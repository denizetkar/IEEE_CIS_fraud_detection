import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import helper
from helper import LargeTabularDataset, DataEpochGenerator
from model_handler import FixedInputFixedOutputModelHandler
from ffn import FeedForwardNN


def main():
    # Define all necessary paths
    model_path = os.path.join('models', 'test_ffn.model')
    category_mapping_path = os.path.join('data', 'test_category_mappings.pickle')
    test_data_path = os.path.join('data', 'test_data.hdf5')
    submission_path = os.path.join('data', 'submission.csv')

    test_transaction_path = os.path.join('data', 'test_transaction.csv')
    test_identity_path = os.path.join('data', 'test_identity.csv')

    # Define categorical columns by hand
    categorical_cols = \
        ['ProductCD'] + \
        ['card' + str(i) for i in range(1, 7)] + \
        ['addr' + str(i) for i in range(1, 3)] + \
        ['P_emaildomain', 'R_emaildomain'] + \
        ['M' + str(i) for i in range(1, 10)] + \
        ['DeviceType'] + \
        ['id_' + str(i) for i in range(12, 39)] + \
        ['Transaction_dow', 'Transaction_hour', 'device_name']
    # Identify ID column
    id_col = 'TransactionID'

    if not os.path.exists(test_data_path):
        # Read training data
        test_transaction = helper.reduce_memory_usage(pd.read_csv(test_transaction_path))
        test_identity = helper.reduce_memory_usage(pd.read_csv(test_identity_path))
        # Merge 'transaction' and 'identity' on 'TransactionID'
        test_data = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
        del test_transaction, test_identity
        # Create useful time features
        # https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
        test_data['Transaction_dow'] = np.floor((test_data['TransactionDT'] / (3600 * 24) - 1) % 7)
        test_data['Transaction_hour'] = np.floor(test_data['TransactionDT'] / 3600) % 24
        # Remove worthless columns (TransactionDT)
        test_data.drop(['TransactionDT'], axis=1, inplace=True)
        # Aggregate certain categorical columns (e.g. )
        test_data.loc[test_data['id_30'].str.contains('Android', na=False), 'id_30'] = 'Android'
        test_data.loc[test_data['id_30'].str.contains('iOS', na=False), 'id_30'] = 'iOS'
        test_data.loc[test_data['id_30'].str.contains('Windows', na=False), 'id_30'] = 'Windows'
        test_data.loc[test_data['id_30'].str.contains('Linux', na=False), 'id_30'] = 'Linux'
        test_data.loc[test_data['id_30'].str.contains('Mac', na=False), 'id_30'] = 'Mac'

        test_data['device_name'] = test_data['DeviceInfo'].str.split('/', expand=True)[0]
        test_data.drop(['DeviceInfo'], axis=1, inplace=True)
        test_data.loc[test_data['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
        test_data.loc[test_data['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
        test_data.loc[test_data['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
        test_data.loc[test_data['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
        test_data.loc[test_data['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
        test_data.loc[test_data['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
        test_data.loc[test_data['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
        test_data.loc[test_data['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
        test_data.loc[test_data['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
        test_data.loc[test_data['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
        test_data.loc[test_data['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
        test_data.loc[test_data['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
        test_data.loc[test_data['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
        test_data.loc[test_data['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
        test_data.loc[test_data['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
        test_data.loc[test_data['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
        test_data.loc[test_data['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

        with open(category_mapping_path, 'rb') as f:
            category_mappings = pickle.load(f)
        test_data.loc[~test_data['device_name'].isin(
            pd.Index(category_mappings['device_name'].first_keys())), 'device_name'] = "Others"

        test_data['id_31'] = test_data['id_31'].str.replace(r' *[\d\.]+', '')
        # TODO: https://www.kaggle.com/artgor/eda-and-models#Feature-engineering

        def category_to_code(column, category_mappings):
            return pd.to_numeric([category_mappings[column.name].forward(x) for x in column], downcast='integer')

        # Turn all categorical columns to code columns
        test_data[categorical_cols] = test_data[categorical_cols].apply(
            lambda col: category_to_code(col, category_mappings), axis=0)

        continuous_cols = [col for col in test_data.columns if col not in categorical_cols + [id_col]]
        test_data_cont = test_data[continuous_cols]
        # Find the mean (mu) and std (sigma) of each continuous column
        cont_mu, cont_sigma = test_data_cont.mean(), test_data_cont.std()
        # Assign N(mu, sigma) to each NAN in each continuous column
        ran = pd.DataFrame(np.random.randn(*test_data_cont.shape),
                           columns=test_data_cont.columns, index=test_data_cont.index, dtype=np.float32)
        ran = cont_sigma * ran + cont_mu
        test_data_cont.update(ran, overwrite=False)
        del ran, cont_mu, cont_sigma
        test_data[continuous_cols] = test_data_cont
        del test_data_cont
        # Save the 'test_data' into a hdf5 file
        test_data.to_hdf(test_data_path, 'data', mode='w', format='table')
        del test_data
    else:
        with open(category_mapping_path, 'rb') as f:
            category_mappings = pickle.load(f)
        all_cols = pd.read_hdf(test_data_path, stop=0).columns.to_list()
        continuous_cols = [col for col in all_cols if col not in categorical_cols + [id_col]]
        del all_cols

    # --------------MODEL HYPER-PARAMETERS ARE ALL BELOW------------------
    model_handler = FixedInputFixedOutputModelHandler(
        FeedForwardNN, nn.CrossEntropyLoss, torch.optim.Adam,
        {'emb_dims': [(lambda cat_dim: (cat_dim, (cat_dim - 1) // 3 + 1))(len(category_mappings[col]))
                      for col in categorical_cols],
         'no_of_cont': len(continuous_cols),
         'lin_layer_sizes': (lambda inp_dim: [4096, 4096, 2048]) \
             (sum(len(category_mappings[col]) for col in categorical_cols) + len(continuous_cols)),
         'output_size': 2, 'emb_dropout': 0.0,
         'lin_layer_dropouts': [0.1, 0.2, 0.2]}, (), optim_args={'lr': 3e-5},
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        l1_regularization_weight=1.0)

    # Load the model first
    model_handler.load(model_path)

    # Chunk size is the number of rows to read from disk at a time !!!!
    batch_size = 6000
    # Create a dataset and chunk loader for eval data as well
    test_dataset = LargeTabularDataset(data_path=test_data_path, cont_cols=continuous_cols,
                                       cat_cols=categorical_cols, output_col=id_col,
                                       chunksize=20 * batch_size, is_hdf=True)
    # 'batch_size' below is the batch size of chunks !!!!!!!!!!!
    test_chunk_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())

    if os.path.exists(submission_path):
        os.remove(submission_path)
    # Create a batch generator
    use_header = True
    for pred, pred_id in model_handler.predict(DataEpochGenerator(test_chunk_loader, batch_size, epoch=1)):
        pred = F.softmax(pred, dim=-1)[:, 1]
        table = pd.DataFrame({'TransactionID': pred_id.cpu(), 'isFraud': pred.cpu()})
        table.to_csv(submission_path, index=False, mode='a', header=use_header)
        use_header = False


if __name__ == '__main__':
    main()
