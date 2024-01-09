import pandas as pd
from utils import feature_impute, rf_fill, make_index, get_onehot_ori
from sklearn.model_selection import train_test_split
from smote import get_balance_dataset, get_balance_dataset_protype
from noise import inject_g_n
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np


def generate_data(path, random, process_flag, isnoise, need_model_filling, needed_model_filling, h, columns_last,
                  noise):
    train = pd.read_excel(path, sheet_name='train')
    val = pd.read_excel(path, sheet_name='val')
    test = pd.read_excel(path, sheet_name='test')
    x_train = train.iloc[:, 0:11]
    y_train = train.iloc[:, 11:13]
    x_val = val.iloc[:, 0:11]
    y_val = val.iloc[:, 11:13]
    x_test = test.iloc[:, 0:11]
    y_test = test.iloc[:, 11:13]

    # Balance the training portion with the protein portion
    temp_df = pd.concat([x_train, y_train], axis=1)

    temp_df = make_index(temp_df)

    train_data_df = pd.DataFrame(temp_df, columns=train.columns)

    process_flag = 0
    if process_flag == 0:
        train_data = get_balance_dataset_protype(train_data_df, 'ProType', random, 1, columns_last)
    elif process_flag == 1:
        train_data = get_balance_dataset(train_data_df, 'MF/UF', random, columns_last)
    elif process_flag == 2:
        train_data = get_balance_dataset(train_data_df, 'MemShape', random, columns_last)
    else:
        train_data = get_balance_dataset(train_data_df, 'Salt', random, columns_last)

    # Split the training set in x and y
    x_train = train_data.iloc[:, 0:11]
    y_train = train_data.iloc[:, 11:13]

    x_test = x_test[x_train.columns]
    x_val = x_val[x_train.columns]

    len_train = x_train.shape[0]
    len_val = x_val.shape[0]
    len_test = x_test.shape[0]
    x_new = pd.concat([x_train, x_val, x_test], axis=0)
    x_new = make_index(x_new)

    x_new = get_onehot_ori(x_new).copy()
    y_train = make_index(y_train)
    if isnoise:

        x_new.iloc[0:len_train, :] = inject_g_n(data=x_new.iloc[0:len_train, :], isfeature=True, noise=noise)
        y_train = inject_g_n(data=y_train, isfeature=False, noise=noise)

    x_new.drop('MF/UF_', axis=1, inplace=True)
    x_new = pd.get_dummies(x_new)
    std = x_new.std()
    mean = x_new.mean()
    x_new = (x_new - mean) / std
    mean, std = 0, 0
    x_train = x_new.iloc[0:len_train, :]
    x_val = x_new.iloc[len_train:len_train + len_val, :]
    x_test = x_new.iloc[len_train + len_val:len_train + len_val + len_test, :]

    x_train = torch.from_numpy(x_train.values).float()
    y_train = torch.from_numpy(y_train.values).float()
    x_test = torch.from_numpy(x_test.values).float()
    y_test = torch.from_numpy(y_test.values).float()
    x_val = torch.from_numpy(x_val.values).float()
    y_val = torch.from_numpy(y_val.values).float()
    print(y_train.shape, y_val.shape, y_test.shape)
    train_data = Data.TensorDataset(x_train, y_train)

    train_load = Data.DataLoader(
        dataset=train_data,
        batch_size=1820,
        shuffle=True,
    )
    print(train_load)

    return mean, std, train_load, x_train, x_val, x_test, y_train, y_val, y_test
