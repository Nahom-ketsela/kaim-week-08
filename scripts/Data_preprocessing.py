import pandas as pd
import numpy as np
import ipaddress
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data files
def load_data(fraud_path, ip_path, credit_path):
    return (
        pd.read_csv(fraud_path),
        pd.read_csv(ip_path),
        pd.read_csv(credit_path)
    )

# Handle missing values by dropping or filling them
def handle_missing_values(df, method='drop', fill_value=None):
    if method == 'drop':
        return df.dropna()
    elif fill_value is not None:
        return df.fillna(fill_value)
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        return df

# Remove duplicate rows
def remove_duplicates(df):
    return df.drop_duplicates()

# Convert columns to datetime or numeric
def correct_data_types(df, datetime_cols=None, numeric_cols=None):
    if datetime_cols:
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col])
    if numeric_cols:
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Convert IP address to integer
def convert_ip_to_int(ip):
    return int(ipaddress.ip_address(ip))

# Merge IP-to-country data with fraud data
def merge_ip_country(df_fraud, df_ip_country):
    df_fraud['ip_int'] = df_fraud['ip_address'].apply(convert_ip_to_int)
    df_ip_country.sort_values('lower_bound_ip_address', inplace=True)
    merged_df = pd.merge_asof(
        df_fraud.sort_values('ip_int'),
        df_ip_country,
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    return merged_df[
        (merged_df['ip_int'] >= merged_df['lower_bound_ip_address']) &
        (merged_df['ip_int'] <= merged_df['upper_bound_ip_address'])
    ]

# Add new features
def feature_engineering(df):
    if 'purchase_time' in df.columns:
        df['purchase_hour'] = df['purchase_time'].dt.hour
        df['purchase_day_of_week'] = df['purchase_time'].dt.dayofweek
    df['user_transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
    return df

# Normalize or scale numeric columns
def normalize_and_scale(df, columns=None, scaler=None):
    if columns and scaler:
        df[columns] = scaler.fit_transform(df[columns])
    return df

# Encode categorical features
def encode_categorical_features(df, cols):
    for col in cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df
