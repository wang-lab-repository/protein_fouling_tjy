import pandas as pd

from utils import mix_seed
from data import generate_data
from model import model
from loss import MyLoss
import torch
import warnings
import numpy as np

warnings.filterwarnings("ignore")

need_model_filling = ['Steady Flux (LMH)', 'Rejection (%)']
needed_model_filling = ['MF/UF', 'MF/UF']  # 'Bond-ing','Loading'
columns_last = ['ProConc (ppm)', 'MemSize (micron)', 'CFV (m/s)', 'TMP (bar)', 'pH',
                'Salt', 'Temp (℃)', 'ProType', 'MemShape', 'Mem Type', 'MF/UF',
                'Steady Flux (LMH)', 'Rejection (%)']
h = []
ymax0 = 7464.00
ymin0 = 0
ymax1 = 100
ymin1 = 0
mix = 166
noise = {'ProConc (ppm)': 40000.0,
         'MemSize (micron)': 1.44,
         'CFV (m/s)': 7.0,
         'TMP (bar)': 7.0,
         'pH': 9.0,
         'Temp (℃)': 50.0,
         'Steady Flux (LMH)': 7464.0,
         'Rejection (%)': 100.0}


def inference():
    mix_seed(mix)  # train_load, x_train, x_test, y_train, y_test

    _, _, train_load, x_train, x_val, x_test, y_train, y_val, y_test = generate_data(path="data/protein_fouling.xlsx",
                                                                                     random=mix,
                                                                                     process_flag=0,
                                                                                     isnoise=True,
                                                                                     need_model_filling=need_model_filling,
                                                                                     needed_model_filling=needed_model_filling,
                                                                                     h=h,
                                                                                     columns_last=columns_last,
                                                                                     noise=noise)

    net = model()
    net.load_state_dict(torch.load('checkpoint/model_params.ckpt'), False)  #
    # training....
    loss_func = MyLoss()
    net.eval()  # Model Freeze
    prediction_test = net(x_test)
    # Calculate R² for each output feature on the test set
    R2_test_1 = 1 - torch.mean((y_test[:, 0] - prediction_test[:, 0]) ** 2) / torch.mean(
        (y_test[:, 0] - torch.mean(y_test[:, 0])) ** 2)
    R2_test_2 = 1 - torch.mean((y_test[:, 1] - prediction_test[:, 1]) ** 2) / torch.mean(
        (y_test[:, 1] - torch.mean(y_test[:, 1])) ** 2)
    # Calculate the NRMSE for each output feature on the test set
    RMSE_test_1 = torch.sqrt(torch.sum((y_test[:, 0] - prediction_test[:, 0]) ** 2) / len(y_test)) / (ymax0 - ymin0)
    RMSE_test_2 = torch.sqrt(torch.sum((y_test[:, 1] - prediction_test[:, 1]) ** 2) / len(y_test)) / (ymax1 - ymin1)
    # Output the final model evaluation results
    print("------------------------结果------------------------")
    print("r2:")
    print(f'test: RWP：{R2_test_1},RSP {R2_test_2}\n')
    print("NRMSE:")
    print(f'test: RWP：{RMSE_test_1},RSP {RMSE_test_2}\n')


def eq(path="data/tensor.npy", x_train=pd.DataFrame()):
    b = np.load(path, allow_pickle=True)
    new = [torch.tensor(x) for x in b]
    x_train_new = new[0].reshape(1, -1)
    for i in range(1, len(new)):
        x_train_new = torch.cat([x_train_new, new[i].reshape(1, -1)], axis=0)
    print(x_train.shape, x_train_new.shape)
    print(torch.sum(x_train == x_train_new))
    print(x_train)
    print(x_train_new)
