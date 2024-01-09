from imblearn.over_sampling import SMOTE
import pandas as pd


def get_balance_dataset(data, column, random, columns_last):

    y_train = data[column]
    x_train = data.drop(column, axis=1)

    smote = SMOTE(random_state=random, k_neighbors=3)


    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    x_train_resampled = pd.DataFrame(x_train_resampled, columns=x_train.columns)
    y_train_resampled = pd.DataFrame(y_train_resampled, columns=y_train.to_frame().columns)
    data_sampled = pd.concat([x_train_resampled, y_train_resampled], axis=1)
    data_sampled = data_sampled[columns_last]
    return data_sampled


def get_balance_dataset_protype(data, column, random, strategy, columns_last):

    data = data.iloc[:, 0:13]
    y_train = data[column]
    x_train = data.drop(column, axis=1)

    if strategy == 0:
        y_train[y_train == 1].fillna(0, axis=0, inplace=True)
    else:
        list = []
        for i in range(x_train.shape[0]):
            if (y_train[i] not in (1, 4, 7, 10)):
                list.append(i)
        x_train.drop(list, axis=0, inplace=True)
        y_train.drop(list, axis=0, inplace=True)

    smote = SMOTE(random_state=random, k_neighbors=3)


    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    x_train_resampled = pd.DataFrame(x_train_resampled, columns=x_train.columns)
    y_train_resampled = pd.DataFrame(y_train_resampled, columns=y_train.to_frame().columns)
    data_sampled = pd.concat([x_train_resampled, y_train_resampled], axis=1)
    data_sampled = data_sampled[columns_last]
    return data_sampled
