import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1 Function to identify numeric and categorical variables
def identify_numeric_categorical(df):
    numeric_features = df.select_dtypes(include=['float', 'int']).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    return numeric_features, categorical_features

# 2 Function to drop duplicate rows
def drop_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

# 3 Function to delete incomplete data based on a threshold
def delete_incomplete_data(df, threshold=0.8):
    return df.dropna(thresh=len(df.columns) * threshold)

# Function to fill missing numerical values with mean
def fill_missing_with_mean(df, numeric_features):
    for feature in numeric_features:
        if feature in df.columns:
            mean_value = df[feature].mean()
            df[feature].fillna(mean_value, inplace=True)
    return df

# Function to fill missing categorical values with mode
def fill_missing_with_mode(df, categorical_features):
    for feature in categorical_features:
        if feature in df.columns:
            mode_value = df[feature].mode()
            if not mode_value.empty:
                mode_value = mode_value.iloc[0]
                df[feature].fillna(mode_value, inplace=True)
    return df



# 4 Function to remove oversampled instances
def remove_oversamples(df, max_occurrences=2, min_to_keep=1):
    groups = df.groupby(df.columns.tolist(), as_index=False)
    
    def sample(group):
        n = max(min_to_keep, min(len(group), max_occurrences))
        if len(group) >= n:
            return group.sample(n)
        else:
            return pd.DataFrame()  # Return an empty DataFrame
        
    sampled_groups = [sample(group) for _, group in groups]
    sampled_groups = [group for group in sampled_groups if not group.empty]  # Remove empty groups
    if sampled_groups:
        filtered_df = pd.concat(sampled_groups, ignore_index=True)
        return filtered_df
    else:
        return df.copy()




# Function to remove incomplete rows based on relevant columns
def remove_incomplete_irrelevant_responses(df, relevant_columns):
    existing_columns = [col for col in relevant_columns if col in df.columns]
    return df.dropna(subset=existing_columns)

def one_hot_encode(df, categorical_features=None):
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object']).columns
        
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    return df_encoded

# Function to detect and handle outliers in numeric features using IQR
def handle_outliers(df, numeric_features, lower_factor=1.5, upper_factor=1.5):
    for feature in numeric_features:
        if feature in df.columns:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (lower_factor * IQR)
            upper_bound = Q3 + (upper_factor * IQR)
            
            df[feature] = np.where((df[feature] < lower_bound) | (df[feature] > upper_bound),
                                   df[feature].mean(), df[feature])
    
    return df

# Function to normalize numeric features using StandardScaler
def normalize_numeric_features(df, numeric_features):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_features])
    df[numeric_features] = scaled_data
    return df

def preprocess_data(df, numeric_features, categorical_features, threshold=0.8, relevant_columns=['Response1', 'Response2']):
    numeric_features, categorical_features = identify_numeric_categorical(df)
    data = drop_duplicates(df)
    data = delete_incomplete_data(data, threshold)
    data = remove_oversamples(data)
    data = fill_missing_with_mean(data, numeric_features)
    data = fill_missing_with_mode(data, categorical_features)
    data = remove_incomplete_irrelevant_responses(data, relevant_columns)
    data = handle_outliers(data, numeric_features)
    data = normalize_numeric_features(data, numeric_features)
    data = one_hot_encode(data, categorical_features)
    return data