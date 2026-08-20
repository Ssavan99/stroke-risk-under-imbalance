import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    data = pd.get_dummies(data=data)
    scaler = StandardScaler()
    con_cols=['age','avg_glucose_level','bmi']
    
    X = data.drop(columns=["id", "stroke"])
    X[con_cols] = scaler.fit_transform(X[con_cols])
    y = data["stroke"].to_list()
    return X,y

def get_smote_data():
    X, y = get_clean_data()
    oversampler = SMOTE()
    X, y = oversampler.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def get_adasyn_data():
    X, y = get_clean_data()
    oversampler = ADASYN()
    X, y = oversampler.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def get_ros_data():
    X, y = get_clean_data()
    oversampler = RandomOverSampler()
    X, y = oversampler.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test