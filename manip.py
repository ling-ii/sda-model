import os
import numpy as np
import numpy.typing as npt
import pandas as pd

def loader(path: str) -> tuple[npt.ArrayLike]:
    
    if not os.path.exists(path):
        raise NotADirectoryError(f"Path '{path}' does not exist.")

    batches = pd.Series(
        os.listdir(path)).apply(lambda batch: os.path.join(path, batch)
    )

    is_dir = (batches.apply(os.path.isdir))
    batches = batches[is_dir]

    proper_element_list = []
    basename_list = []

    for batch in batches:
        parents = [os.path.join(batch, parent) for parent in os.listdir(batch)]
        for parent in parents:
            fragments = [os.path.join(parent, fragment) for fragment in os.listdir(parent)]
            for fragment in fragments:
                basename_list.append(os.path.basename(fragment))
                proper_element_list.append(pd.read_csv(fragment)['r'])

    proper_element_array = np.asarray(proper_element_list, dtype=np.float32)
    basename_array = np.asarray(basename_list, dtype=str)

    return proper_element_array, basename_array    

def drop_nan(data:npt.ArrayLike, labels:npt.ArrayLike) -> tuple[npt.ArrayLike]:
    nan_mask = np.isnan(data)
    row_mask = np.any(nan_mask, axis=1)

    return data[~row_mask], labels[~row_mask]

def generate_pairs(data:npt.ArrayLike, labels:npt.ArrayLike) -> npt.ArrayLike:
    # Generate numpy array of parent labels
    labels = pd.Series(labels).str.findall(r'[0-9]+').str[0].to_numpy()

    n = data.shape[0]
    i, j = np.tril_indices(n, k=-1)
    data_i = data[i]
    data_j = data[j]
    is_same_label = (labels[i] == labels[j]).astype(int)

    return np.column_stack([data_i, data_j, is_same_label])

def shuffle_balance_data(data: npt.ArrayLike) -> npt.ArrayLike:

    is_true = (data[:, -1] == 1)
    data_t = (data[is_true])
    data_f = (data[np.logical_not(is_true)])
    
    np.random.default_rng(3489756).shuffle(data_t)
    np.random.default_rng(2384675).shuffle(data_f)

    ratio = 1 / (len(data_t) / len(data_f))

    return np.concatenate((data_t, data_f[::int(ratio)]))
