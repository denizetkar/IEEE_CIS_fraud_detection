import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import helper
from helper import BiDirectionalDict, LargeTabularDataset, DataEpochGenerator
from model_handler import FixedInputFixedOutputModelHandler
from ffn import FeedForwardNN


def main():
    # Define all necessary paths
    model_path = os.path.join('models', 'ffn.model')
    category_mapping_path = os.path.join('data', 'category_mappings.pickle')
    train_eval_data_path = os.path.join('data', 'train_eval_data.hdf5')
    train_data_path = os.path.join('data', 'train_data.hdf5')
    eval_data_path = os.path.join('data', 'eval_data.hdf5')

    train_transaction_path = os.path.join('data', 'train_transaction.csv')
    train_identity_path = os.path.join('data', 'train_identity.csv')

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
    # Identify output column
    output_col = 'isFraud'

    if not os.path.exists(train_eval_data_path) or not os.path.exists(category_mapping_path):
        # Read training data
        train_transaction = helper.reduce_memory_usage(
            pd.read_csv(train_transaction_path))
        train_identity = helper.reduce_memory_usage(
            pd.read_csv(train_identity_path))
        # Merge 'transaction' and 'identity' on 'TransactionID'
        train_eval_data = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
        del train_transaction, train_identity
        # Create useful time features
        # https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
        train_eval_data['Transaction_dow'] = np.floor((train_eval_data['TransactionDT'] / (3600 * 24) - 1) % 7)
        train_eval_data['Transaction_hour'] = np.floor(train_eval_data['TransactionDT'] / 3600) % 24
        # Remove worthless columns (TransactionID, TransactionDT)
        train_eval_data.drop(['TransactionID', 'TransactionDT'], axis=1, inplace=True)
        # Aggregate certain categorical columns (e.g. )
        train_eval_data.loc[train_eval_data['id_30'].str.contains('Android', na=False), 'id_30'] = 'Android'
        train_eval_data.loc[train_eval_data['id_30'].str.contains('iOS', na=False), 'id_30'] = 'iOS'
        train_eval_data.loc[train_eval_data['id_30'].str.contains('Windows', na=False), 'id_30'] = 'Windows'
        train_eval_data.loc[train_eval_data['id_30'].str.contains('Linux', na=False), 'id_30'] = 'Linux'
        train_eval_data.loc[train_eval_data['id_30'].str.contains('Mac', na=False), 'id_30'] = 'Mac'

        train_eval_data['device_name'] = train_eval_data['DeviceInfo'].str.split('/', expand=True)[0]
        train_eval_data.drop(['DeviceInfo'], axis=1, inplace=True)
        train_eval_data.loc[train_eval_data['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
        train_eval_data.loc[train_eval_data['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'
        train_eval_data.loc[train_eval_data['device_name'].isin(
            train_eval_data['device_name'].value_counts()[train_eval_data['device_name'].value_counts() < 200].index),
                            'device_name'] = "Others"

        train_eval_data['id_31'] = train_eval_data['id_31'].str.replace(r' *[\d\.]+', '')
        # TODO: https://www.kaggle.com/artgor/eda-and-models#Feature-engineering
        # Cast all categorical columns to 'category' data type
        train_eval_data[categorical_cols] = \
            train_eval_data[categorical_cols].astype('category')
        # Get numerical mapping between categories and codes for each categorical column
        category_mappings = {}
        for col in categorical_cols:
            category_mapping = BiDirectionalDict({helper.nan: 0})
            category_mapping.update(
                {category: code + 1
                 for code, category in enumerate(train_eval_data[col].cat.categories)})
            category_mappings[col] = category_mapping
        helper.safe_del(['col', 'category_mapping'], locals())
        # Save the category mappings
        with open(category_mapping_path, 'wb') as f:
            pickle.dump(category_mappings, f)

        # Turn all categorical columns to code columns
        def category_to_code(column):
            return column.cat.codes + 1

        def code_to_category(column, category_mappings):
            return column.apply(lambda x: category_mappings[column.name].backward(x))

        train_eval_data[categorical_cols] = train_eval_data[categorical_cols].apply(category_to_code)

        continuous_cols = [col for col in train_eval_data.columns if col not in categorical_cols + [output_col]]
        train_data_cont = train_eval_data[continuous_cols]
        # Find the mean (mu) and std (sigma) of each continuous column
        cont_mu, cont_sigma = train_data_cont.mean(), train_data_cont.std()
        # Assign N(mu, sigma) to each NAN in each continuous column
        ran = pd.DataFrame(np.random.randn(*train_data_cont.shape),
                           columns=train_data_cont.columns, index=train_data_cont.index, dtype=np.float32)
        ran = cont_sigma * ran + cont_mu
        train_data_cont.update(ran, overwrite=False)
        del ran, cont_mu, cont_sigma
        train_eval_data[continuous_cols] = train_data_cont
        del train_data_cont
        # Save the 'train_eval_data' into a hdf5 file
        train_eval_data.to_hdf(train_eval_data_path, 'data', mode='w', format='table')
        del train_eval_data
    else:
        with open(category_mapping_path, 'rb') as f:
            category_mappings = pickle.load(f)
        all_cols = pd.read_hdf(train_eval_data_path, stop=0).columns.to_list()
        continuous_cols = [col for col in all_cols if col not in categorical_cols + [output_col]]
        del all_cols

    # Split the whole data into training and evaluation data
    if not os.path.exists(train_data_path) or not os.path.exists(eval_data_path):
        helper.train_eval_split_hdf5(train_eval_data_path, train_data_path, eval_data_path)

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
    epoch = 10
    train_dataset = LargeTabularDataset(data_path=train_data_path, cont_cols=continuous_cols,
                                        cat_cols=categorical_cols, output_col=output_col,
                                        chunksize=24 * batch_size, shuffle=True, is_hdf=True)
    # 'batch_size' below is the batch size of chunks !!!!!!!!!!!
    train_chunk_loader = DataLoader(train_dataset, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())

    # Create a dataset and chunk loader for eval data as well
    eval_dataset = LargeTabularDataset(data_path=eval_data_path, cont_cols=continuous_cols,
                                       cat_cols=categorical_cols, output_col=output_col,
                                       chunksize=2 * batch_size, is_hdf=True)
    # 'batch_size' below is the batch size of chunks !!!!!!!!!!!
    eval_chunk_loader = DataLoader(eval_dataset, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())

    # Create a batch generator
    step_per_update = 20
    update_per_verbose = 2
    batch_per_epoch = (train_dataset.nb_samples - 1) // batch_size + 1
    update_per_epoch = (batch_per_epoch - 1) // step_per_update + 1
    model_handler.train(in_tgt_generator=DataEpochGenerator(train_chunk_loader, batch_size, epoch=epoch),
                        eval_in_tgt_generator=DataEpochGenerator(eval_chunk_loader, batch_size, epoch=1),
                        update_per_eval=update_per_epoch, step_per_update=step_per_update,
                        update_per_verbose=update_per_verbose, summary_func_list=[helper.l2_norm_model])
    model_handler.save(model_path)
    pass


if __name__ == '__main__':
    main()
