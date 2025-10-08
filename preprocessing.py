from sklearn import preprocessing
import numpy as np

def process_data(data):

    # Eccentricity    -> (0, inf)
    # Inclination     -> [0, 2*pi)
    # Semi-major axis -> (~6.371e6, inf)
    preprocessors = [
        preprocessing.RobustScaler,
        preprocessing.MinMaxScaler,
        preprocessing.RobustScaler,
        preprocessing.RobustScaler,
        preprocessing.MinMaxScaler,
        preprocessing.RobustScaler,
    ]

    # TODO include scalers and data for PNC and QTN
    selector = np.r_[0:3, 9:12]
    mee_data = data[:, selector]

    datasets = [mee_data[:, i].reshape(-1, 1) for i in range(6)]

    scaling_sets = [
        datasets[0],
        np.linspace(0, 2*np.pi, 2).reshape(-1, 1),
        datasets[2],
        datasets[3],
        np.linspace(0, 2*np.pi, 2).reshape(-1, 1),
        datasets[5]
    ]

    scalers = [scaler().fit(ds) for scaler, ds in zip(preprocessors, scaling_sets)]
    transformed_datasets = [scaler.transform(ds) for scaler, ds in zip(scalers, datasets)]

    return np.column_stack(transformed_datasets)