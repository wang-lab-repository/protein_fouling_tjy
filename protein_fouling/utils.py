import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import os
import random
import torch


def feature_impute(df_all, target, use_f):
    df_mv = df_all[[use_f, target]].copy()
    df_mv = df_mv.dropna()
    x = np.array(df_mv[use_f]).reshape(-1, 1)
    y = np.array(df_mv[target]).reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    scores = cross_val_score(rf, x_train, y_train, cv=3, scoring='r2')
    score = np.mean(scores)
    name = target + "_imputed"
    p = rf.predict(np.array(df_all[use_f]).reshape(-1, 1))
    df_all[name] = p
    return df_all, rf


def feature_impute_exiting(df_all, target, use_f, model):
    name = target + "_imputed"
    p = model.predict(np.array(df_all[use_f]).reshape(-1, 1))
    df_all[name] = p
    return df_all


def rf_fill(df_all, target):
    df_all[target] = df_all[target].fillna(df_all[target + "_imputed"])
    # drop unrequired featues
    df = df_all.drop([target + "_imputed"], axis=1, inplace=True)


def generate_o_d(origin, str, data):
    list = []
    for i in range(data.shape[0]):
        list.append(origin[data.iloc[i][str]])
    data[str + '_'] = list
    data.drop(str, axis=1, inplace=True)
    return data


def make_index(data):
    index = []
    for i in range(data.shape[0]):
        index.append(i)
    data.index = index
    return data


# 'ProType', 'MF/UF', 'MemShape', 'Salt'
def get_onehot_ori(data):
    ProType = {1: 'BSA', 2: 'β-lactoglobulin', 3: 'IgG', 4: 'Pepsin', 5: 'Invertase', 6: 'Lipase', 7: 'Hemoglobin',
               8: 'Myoglobin', 9: 'Lysozyme', 10: 'α-Lactalbumin', 11: 'ovalbumin', 12: 'Casein', 13: 'Trypsin',
               14: 'Egg Albumin'}

    MemShape = {0: 'Flat', 1: 'Tubular', 2: 'HF'}

    MF_UF = {0: 'MF', 1: 'UF'}

    Salt = {0: 'no', 1: 'yes'}

    MemType = {1: 'Psf', 2: 'mod Psf', 3: 'PVDF', 4: 'mod PVDF', 5: 'PES', 6: 'mod PES', 7: 'PAN', 8: 'mod PAN',
               9: 'PC', 10: 'mod PC', 11: 'Cellulose', 12: 'mod Cellulose',
               13: 'PTFE', 14: 'mod PTFE', 15: 'PEI', 16: 'mod PEI', 17: 'PI', 18: 'mod PI', 19: 'PVC', 20: 'mod PVC',
               21: 'PEES', 22: 'mod PEES', 23: 'Al Oxide', 24: 'Titania',
               25: 'Zirconia', 26: 'SPG', 27: 'mod SPG', 28: 'ZR-TI', 29: 'Ceramic', 30: 'PET', 31: 'Polyester',
               32: 'TFC PA', 33: 'EVOH'}

    data = generate_o_d(ProType, 'ProType', data)
    data = generate_o_d(MemShape, 'MemShape', data)
    data = generate_o_d(MemType, 'Mem Type', data)
    data = generate_o_d(MF_UF, 'MF/UF', data)
    data = generate_o_d(Salt, 'Salt', data)
    return data


def mix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
