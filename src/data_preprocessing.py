
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_data(path):
    df = pd.read_csv(path)
    if 'Time' in df.columns:
        df = df.sort_values('Time')
    return df

def train_test_time_split(df, test_size=0.2):
    # Use a time-based split (past -> train, future -> test) to avoid leakage from random shuffling
    n_test = int(len(df) * test_size)
    train_df = df.iloc[:-n_test]
    test_df = df.iloc[-n_test:]
    return train_df, test_df

def build_preprocess_pipeline():
    scaler = StandardScaler()
    pipe = Pipeline([
        ('scaler', scaler)
    ])
    return pipe

def prepare_features(df):
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y
