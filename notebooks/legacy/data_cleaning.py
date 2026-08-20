import pandas as pd


def convert_to_number(df, col_name):
    uniques = df[col_name].unique()

    map_uniques = {}
    for i, x in enumerate(uniques):
        map_uniques[x] = i
    df[col_name] = df[col_name].apply(lambda x: map_uniques[x])
    return df


def get_clean_data():
    data = pd.read_csv("healthcare-dataset-stroke-data.csv")
    data = data.dropna()
    columns = data.columns
    for col_name in columns:
        if (data[col_name].dtype == object):
            data = convert_to_number(data, col_name=col_name)
    X = data.drop(columns=["id", "stroke"])
    y = data["stroke"].to_list()
    return X, y
