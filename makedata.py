import manip
import preprocessing
import os
import numpy as np

def main():

    # Data load, cleaning, and generate pairs
    path = './data/'
    print(f"Loading dataset from '{path}' ...")
    data, basenames = manip.loader(path)
    data, basenames = manip.drop_nan(data, basenames)
    pairs = manip.generate_pairs(data, basenames)
    pairs = manip.shuffle_balance_data(pairs)

    # Preprocess
    print(f"Preprocessing data ...")
    pair_data = pairs[:, :-1]
    pair_data = preprocessing.process_data(pair_data)
    pair_labels = pairs[:, -1]

    # Reconnect data and labels
    pairs = np.column_stack((pair_data, pair_labels))
    print(f"Data type: {pairs.dtype}")

    # Save data to file
    print(f"Saving ...")
    np.save(os.path.join(path, 'dataset'), pairs)
    print(f"Done.")

    return

if __name__ == '__main__':
    main()