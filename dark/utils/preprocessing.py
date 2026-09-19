import os
import glob
import h5py
import pandas as pd
import numpy as np


def preprocessing(root='data/raw_training_files', num_points=1024):
    """
    Load all .h5 files, combine the data, and return them as NumPy arrays.
    """

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, root)

    all_data = []
    all_label = []

    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    if not csv_files:
        raise ValueError("No CSV files found in this data directory")

    # Convert CSV files into HDF5 format
    for file_path in csv_files:
        df = pd.read_csv(file_path)

        # Extract point cloud data and labels
        data = df[['x', 'y', 'z']].values
        label = df['label'].values

        # Add them to data
        all_data.append(data)
        all_label.append(label)
        num_samples = len(data) // num_points
        reshaped_data = data[:num_samples *
                             num_points].reshape(num_samples, num_points, 3)
        reshaped_labels = label[:num_samples *
                                num_points].reshape(num_samples, num_points)

        # Generate HDF5 file path
        h5_path = file_path.replace('.csv', '.h5')

        # Save data in HDF5 format
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('data', data=reshaped_data.astype(
                'float32'))  # Point clouds
            f.create_dataset('label', data=reshaped_labels[:, 0].astype(
                'int64'))  # Labels (1 per cloud)

        print(f"Filtered point clouds saved in HDF5 format: {h5_path}")


if __name__ == '__main__':
    preprocessing()
