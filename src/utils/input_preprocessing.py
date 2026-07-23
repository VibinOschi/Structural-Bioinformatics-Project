import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder


def get_label_encoder_from_dataframe(label_column):
    label_encoder = LabelEncoder()
    label_encoder.fit(label_column)
    return label_encoder


def preprocess_data_files_from_path(path, feature_columns, augment_duplicate=False):
    df = combine_multiple_tsv_into_dataframe(path)
    df = drop_lines_with_empty_values(df, feature_columns)
    if augment_duplicate:
        df = rename_empty_labels(df)
        return duplicate_and_invert_source_and_target_columns(df, feature_columns)
    else:
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


def duplicate_and_invert_source_and_target_columns(dataframe, feature_columns):
    assert isinstance(feature_columns, list) and all(isinstance(col, str) and col.startswith(("s_", "t_")) for col in feature_columns), "feature_columns must be a list of strings starting with 's_' or 't_'"

    duplicate_df = dataframe.copy()

    s_cols = [col for col in feature_columns if col.startswith("s_")]
    for s_col in s_cols:
        suffix = s_col[2:]
        t_col = "t_" + suffix
        if t_col in feature_columns:
            duplicate_df[s_col] = dataframe[t_col]
            duplicate_df[t_col] = dataframe[s_col]

    joined_dataframes = pd.concat([duplicate_df, dataframe], ignore_index=True)

    return joined_dataframes