import pandas as pd
import os

def preprocess_data_files_from_path(path, feature_columns):
    df = combine_multiple_tsv_into_dataframe(path)
    df = drop_lines_with_empty_values(df, feature_columns)
    return rename_empty_labels(df)

def combine_multiple_tsv_into_dataframe(path):
    dataframes = []

    for filename in os.listdir(path):
        dataframes.append(pd.read_csv(str(os.path.join(path, filename)), sep='\t', encoding='latin-1'))

    return pd.concat(dataframes)


def drop_lines_with_empty_values(dataframe, feature_columns):
    return dataframe.dropna(subset=feature_columns)


def rename_empty_labels(dataframe):
    dataframe['Interaction'] = dataframe['Interaction'].fillna('Missing')
    return dataframe