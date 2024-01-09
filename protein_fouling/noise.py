import numpy as np


def inject_g_n(data, isfeature, noise):
    if isfeature:
        columns = ['ProConc (ppm)', 'MemSize (micron)', 'CFV (m/s)', 'TMP (bar)', 'pH', 'Temp (℃)']
    else:
        columns = ['Steady Flux (LMH)', 'Rejection (%)']
    for i in range(data.shape[0]):
        for c in columns:
            temp = np.random.randn(1, 1)
            data.loc[i, c] = data.loc[i, c] + temp[0] / noise[c] / 10
    return data
