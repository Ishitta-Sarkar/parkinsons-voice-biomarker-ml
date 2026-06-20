import pandas as pd


def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset


def dataset_summary(dataset):
    print("Rows:", dataset.shape[0])
    print("Columns:", dataset.shape[1])

    print("\nColumn Names:")
    print(dataset.columns.tolist())
