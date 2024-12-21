import os
import sys

# Go up one directory from the current script's directory -> necessary for imports
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PROJECT_PATH)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm.sklearn import LGBMRegressor
from sklearn import metrics
from sklearn.model_selection import KFold
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from preprocessing.preprocessing_utils import set_invalids_to_nan
from onnxmltools.convert import convert_lightgbm
import onnx
from skl2onnx.common.data_types import FloatTensorType

def create_data_sets(df, target_name):
    """
    Splits the given dataframe into separate feature sets based on column prefixes and a target column.

    This function identifies columns that have prefixes 'gps_' and 'gyro_' to create
    two separate datasets. It also retrieves the target dataset using the provided target name.

    Parameters:
    - df (pd.DataFrame): The source dataframe from which feature sets and target set are derived.
    - target_name (str): The name of the target column in the source dataframe.

    Returns:
    - tuple: A tuple containing three dataframes:
        - X_gps (pd.DataFrame): Dataframe containing features with the 'gps_' prefix.
        - X_gyro (pd.DataFrame): Dataframe containing features with the 'gyro_' prefix.
        - y (pd.DataFrame or pd.Series): The target dataset derived from the column named `target_name`.
    """

    # get column names
    gps_features = list(df.columns[df.columns.str.contains('gps_')])
    gyro_features = list(df.columns[df.columns.str.contains('gyro_')])

    # get X and y
    X_gps = df[gps_features]
    X_gyro = df[gyro_features]
    y = df[target_name]

    return X_gps, X_gyro, y


def create_X_meta(y_pred_dict, df):
    """
    Creates a meta dataframe using predictions and validation columns.

    This function takes in a dictionary of predictions and a source dataframe.
    The resulting meta dataframe contains prediction columns and their corresponding validation columns.

    Parameters:
    - y_pred_dict (dict): A dictionary where keys are prediction column names
      and values are lists or arrays of predictions.
    - df (pd.DataFrame): The source dataframe that must contain validation columns
      corresponding to the prediction columns in y_pred_dict.

    Returns:
    - X_meta (pd.DataFrame): A dataframe containing prediction columns from y_pred_dict
      and their associated validation columns from df.
    """

    # Validate input lengths
    for key in y_pred_dict:
        assert len(y_pred_dict[key]) == len(df), f"Length mismatch: {key} and df must have the same length."

    # Validate dataframe length and columns
    valid_cols = set([key.split('_')[0] + '_valid' for key in y_pred_dict.keys()])
    assert all(col in df.columns for col in
               valid_cols), f"Missing columns in df: {', '.join(valid_cols)} columns must be present."

    # Create X_meta dataframe
    X_meta = pd.DataFrame(y_pred_dict)

    # Add valid columns from df to X_meta
    for col in valid_cols:
        X_meta[col] = df[col].values

    return X_meta


def generate_missing_values(data, exist_prob, na_whole_trips=True, chunk_size=30):
    """
    Generates synthetic missing values (NaNs) in the data based on specified conditions.

    The function allows for two modes:
    1. Introducing missing values for sensors on entire trips.
    2. Introducing missing values in chunks of a specified duration.

    Parameters:
    - data (pd.DataFrame): The original dataframe containing the data for which missing values are to be introduced.
    - exist_prob (float): The probability of a sensor value being present (between 0 and 1). The complement
      (1 - exist_prob) is the probability of the value being missing.
    - na_whole_trips (bool, optional): If True, missing values are introduced for entire trips. If False, missing values
      are introduced in chunks of size `chunk_size`. Defaults to True.
    - chunk_size (int, optional): If `na_whole_trips` is False, this parameter defines the duration in seconds for which
      consecutive missing values will be introduced. Defaults to 30.

    Returns:
    - pd.DataFrame: A dataframe with introduced synthetic missing values based on the specified conditions.

    Notes:
    - The columns 'gps_valid', 'gyro_valid', and 'accel_valid' in the input dataframe are used as indicators for the
      validity of sensor values. A value of 0 indicates missing data (NaN) and a value of 1 indicates valid data.
    - In the case of 'gyro_valid' and 'accel_valid', missing values are introduced simultaneously, i.e., if 'gyro_valid'
      is 0 for a row, 'accel_valid' will also be 0 for that row.
    """
    data = data.copy()

    # create missing sensor value for a whole trip
    if na_whole_trips:
        unique_ids = data['driveID'].unique()
        np.random.RandomState(42).shuffle(unique_ids)
        df_driveIDs = pd.DataFrame(data={'driveID': unique_ids})

        gps_val = np.random.choice([1, 0], size=len(df_driveIDs),
                                                    p=[exist_prob, 1 - exist_prob])
        gyro_accel_val = np.random.choice([1, 0], size=len(df_driveIDs),
                                   p=[exist_prob, 1 - exist_prob])

        # gyro and accel drop out at the same time
        df_driveIDs['gps_valid'] = gps_val
        df_driveIDs['gyro_valid'] = gyro_accel_val
        df_driveIDs['accel_valid'] = gyro_accel_val

        # save column order
        col = data.columns

        # drop actual values
        data = data.drop(columns=['gps_valid', 'gyro_valid', 'accel_valid'])
        data = pd.merge(data, right=df_driveIDs, on='driveID', how='left')

        # restore original column order
        data = data[col]

    # create missing values for chunks of chunk_size seconds
    else:
        n = len(data)  # array length

        # Number of chunks and remainder
        chunks, remainder = divmod(n, chunk_size)

        # Generate random numbers and threshold based on probability
        gps_rand_nums = np.random.rand(chunks + bool(remainder)) < exist_prob
        gyro_accel_rand_nums = np.random.rand(chunks + bool(remainder)) < exist_prob

        # Repeat values for chunks and trim array to required length
        gps_val = np.repeat(gps_rand_nums, chunk_size)[:n]
        gyro_accel_val = np.repeat(gyro_accel_rand_nums, chunk_size)[:n]

        data['gps_valid'] = gps_val
        data['gyro_valid'] = gyro_accel_val
        data['accel_valid'] = gyro_accel_val

    # the value for rows which have an invalid tag should be set to nan.
    data = set_invalids_to_nan(data)

    return data


def set_tags_to_invalid(data, consider_accel=True, consider_gyro=True, consider_gps=True):
    """
    Modifies the validity tags in the data to set specific sensor readings as invalid.

    Based on the parameters provided, this function adjusts the 'valid' columns
    ('gyro_valid', 'accel_valid', 'gps_valid') to indicate invalid readings. After
    setting the tags, the actual sensor values for those marked as invalid will be
    replaced with NaN values using the `set_invalids_to_nan` function.

    Parameters:
    - data (pd.DataFrame): The input dataframe containing sensor readings and their associated validity tags.
    - consider_accel (bool, optional): If False, the 'accel_valid' column will be set to 0 (invalid) for all rows.
                                      Defaults to True.
    - consider_gyro (bool, optional): If False, the 'gyro_valid' column will be set to 0 (invalid) for all rows.
                                     Defaults to True.
    - consider_gps (bool, optional): If False, the 'gps_valid' column will be set to 0 (invalid) for all rows.
                                    Defaults to True.

    Returns:
    - pd.DataFrame: A modified dataframe with updated validity tags and corresponding NaN values for invalid readings.
    """

    if consider_gyro is False:
        data.loc[:, 'gyro_valid'] = 0
    if consider_accel is False:
        data.loc[:, 'accel_valid'] = 0
    if consider_gps is False:
        data.loc[:, 'gps_valid'] = 0

    # the value for rows which have an invalid tag should be set to nan.
    data = set_invalids_to_nan(data)

    return data


def split_randomly(df, generate_nans=False, consider_accel=True, consider_gyro=True, consider_gps=True,
                   train_size=0.9, base_learner_size=0.8, exist_prob_train=1, exist_prob_test=1, na_whole_trips=True, chunk_size=30):
    """
    Splits the data into base-train, meta-train, and test datasets with options to simulate missing values.

    This function randomly divides the dataset based on the unique 'driveID'. It then has options to
    generate synthetic missing values, and mark certain sensor readings as invalid based on the provided flags.
    The final datasets are shuffled and returned.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - generate_nans (bool): If True, simulates missing values based on the given parameters.
    - consider_accel (bool): If False, marks all accelerometer readings as invalid.
    - consider_gyro (bool): If False, marks all gyroscope readings as invalid.
    - consider_gps (bool): If False, marks all GPS readings as invalid.
    - train_size (float): Proportion of data to be used for training (both base and meta combined).
    - base_learner_size (float): Proportion of the train dataset to be used for base learner training.
    - exist_prob_train (float): Probability of existence of a data point in the training set.
    - exist_prob_test (float): Probability of existence of a data point in the test set.
    - na_whole_trips (bool): If True, simulates missing values for entire trips.
    - chunk_size (int): Size of chunks for which missing values will be simulated.

    Returns:
    - df_train (pd.DataFrame): Training dataset for base learners.
    - df_train_meta (pd.DataFrame): Training dataset for meta learner.
    - df_test (pd.DataFrame): Test dataset.
    """

    unique_ids = df['driveID'].unique()
    np.random.RandomState(42).shuffle(unique_ids)

    idx_split_test = int(len(unique_ids) * train_size)
    idx_split_train_base = int(len(unique_ids) * (train_size * base_learner_size))
    train_ids = unique_ids[:idx_split_train_base]
    train_meta_ids = unique_ids[idx_split_train_base:idx_split_test]
    test_ids = unique_ids[idx_split_test:]

    # shuffle training data
    df_train = df[df['driveID'].isin(train_ids)]
    df_train_meta = df[df['driveID'].isin(train_meta_ids)]
    df_test = df[df['driveID'].isin(test_ids)]

    # generate missing values for training data
    if generate_nans:
        df_train = generate_missing_values(df_train, exist_prob=exist_prob_train, na_whole_trips=na_whole_trips, chunk_size=chunk_size)
        df_train_meta = generate_missing_values(df_train_meta, exist_prob=exist_prob_train, na_whole_trips=na_whole_trips, chunk_size=chunk_size)
        df_test = generate_missing_values(df_test, exist_prob=exist_prob_test, na_whole_trips=na_whole_trips, chunk_size=chunk_size)

    # set tags to invalid if defined
    df_train = set_tags_to_invalid(df_train, consider_accel, consider_gyro, consider_gps)
    df_train_meta = set_tags_to_invalid(df_train_meta, consider_accel, consider_gyro, consider_gps)
    df_test = set_tags_to_invalid(df_test, consider_accel, consider_gyro, consider_gps)

    # shuffle training data
    df_train = df_train.sample(frac=1, random_state=42)
    df_train_meta = df_train_meta.sample(frac=1, random_state=42)

    return df_train, df_train_meta, df_test


def split_groupIDs(df, test_groupID, generate_nans=False, consider_accel=True, consider_gyro=True, consider_gps=True,
                   base_learner_size=0.8, exist_prob_train=1, exist_prob_test=1, na_whole_trips=True, chunk_size=30):
    """
    Splits the data based on predefined groupIDs (different driving situations) into train, meta-train, and test datasets with options to simulate missing values.

    Parameters:
    - df (pd.DataFrame): The input dataframe with a 'groupID' column.
    - test_groupID (int): GroupID which will be used exclusively for testing.
    - generate_nans (bool): If True, simulates missing values based on the given parameters.
    - consider_accel (bool): If False, marks all accelerometer readings as invalid.
    - consider_gyro (bool): If False, marks all gyroscope readings as invalid.
    - consider_gps (bool): If False, marks all GPS readings as invalid.
    - base_learner_size (float): Proportion of the training dataset to be used for base learner training.
    - exist_prob_train (float): Probability of existence of a data point in the training set.
    - exist_prob_test (float): Probability of existence of a data point in the test set.
    - na_whole_trips (bool): If True, simulates missing values for entire trips.
    - chunk_size (int): Size of chunks for which missing values will be simulated.

    Returns:
    - df_train (pd.DataFrame): Training dataset for base learners.
    - df_train_meta (pd.DataFrame): Training dataset for meta learner.
    - df_test (pd.DataFrame): Test dataset.

    Notes:
    - This function splits data into training and test datasets based on the provided 'groupID' and 'test_groupID'.
    - The training data is further split into datasets for base learners and meta learners.
    - If 'generate_nans' is True, synthetic missing values are generated based on the provided parameters.
    - Invalid sensor readings are marked based on 'consider_accel', 'consider_gyro', and 'consider_gps'.
    - The training data is then shuffled before returning.
    """

    groups = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_groupIDs = list(set(groups) - {test_groupID})

    # create train splits for base and meta learner
    df_train_all = df[df['groupID'].isin(train_groupIDs)].sample(frac=1, random_state=42)
    unique_ids = df_train_all['driveID'].unique()
    np.random.RandomState(42).shuffle(unique_ids)
    idx_split_train_base = int(len(unique_ids) * base_learner_size)
    train_ids = unique_ids[:idx_split_train_base]
    train_meta_ids = unique_ids[idx_split_train_base:]

    # shuffle training data
    df_train = df[df['driveID'].isin(train_ids)].sample(frac=1, random_state=42)
    df_train_meta = df[df['driveID'].isin(train_meta_ids)].sample(frac=1, random_state=42)
    df_test = df[df['groupID'].isin([test_groupID])]

    # generate missing values for training data
    if generate_nans:
        df_train = generate_missing_values(df_train, exist_prob=exist_prob_train, na_whole_trips=na_whole_trips, chunk_size=chunk_size)
        df_train_meta = generate_missing_values(df_train_meta, exist_prob=exist_prob_train, na_whole_trips=na_whole_trips, chunk_size=chunk_size)
        df_test = generate_missing_values(df_test, exist_prob=exist_prob_test, na_whole_trips=na_whole_trips, chunk_size=chunk_size)

    # set tags to invalid
    df_train = set_tags_to_invalid(df_train, consider_accel, consider_gyro, consider_gps)
    df_train_meta = set_tags_to_invalid(df_train_meta, consider_accel, consider_gyro, consider_gps)
    df_test = set_tags_to_invalid(df_test, consider_accel, consider_gyro, consider_gps)

    return df_train, df_train_meta, df_test


def kfold_cv(df, target_name, folds=5, generate_nans=False, consider_accel=True, consider_gyro=True, consider_gps=True,
             base_learner_size=0.8, exist_prob_train=1, exist_prob_test=1, random_state=42, weighted=True, na_whole_trips=True, chunk_size=30):
    """
    Performs k-fold cross-validation on the given dataframe, simulating missing values and adjusting for sensor validity if required.

    Parameters:
    - df (pd.DataFrame): The input dataframe with a 'driveID' column.
    - target_name (str): Name of the target column.
    - folds (int): Number of cross-validation folds.
    - generate_nans (bool): If True, simulates missing values based on given parameters.
    - consider_accel (bool): If False, marks all accelerometer readings as invalid.
    - consider_gyro (bool): If False, marks all gyroscope readings as invalid.
    - consider_gps (bool): If False, marks all GPS readings as invalid.
    - base_learner_size (float): Proportion of the training dataset used for base learner training.
    - exist_prob_train (float): Probability of a data point's existence in the training set.
    - exist_prob_test (float): Probability of a data point's existence in the test set.
    - random_state (int): Seed value for reproducibility.
    - weighted (bool): If True, it also trains a weighted model for gps and gyro.
    - na_whole_trips (bool): If True, simulates missing values for entire trips.
    - chunk_size (int): Size of chunks for which missing values will be simulated.

    Returns:
    - results (pd.DataFrame): A dataframe containing the evaluation metrics for each fold.
    """

    # store results
    results = pd.DataFrame()

    # get kfolds based on driveIDs
    unique_ids = df['driveID'].unique()
    kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    for i, (all_train_ids, test_ids) in enumerate(kfold.split(unique_ids)):
        print(f'fold {i + 1} out of {folds}')

        # split train_ids for base and meta learner. 70% for training base learner, 30% for meta
        idx_split_train_base = int(len(all_train_ids) * base_learner_size)
        train_ids = all_train_ids[:idx_split_train_base]
        train_meta_ids = all_train_ids[idx_split_train_base:]

        # Split the data into train and test sets and shuffle training data
        df_train = df[df['driveID'].isin(train_ids)].sample(frac=1, random_state=42)
        df_train_meta = df[df['driveID'].isin(train_meta_ids)].sample(frac=1, random_state=42)
        df_test = df[df['driveID'].isin(test_ids)]

        # generate missing values for training data
        if generate_nans:
            df_train = generate_missing_values(df_train, exist_prob=exist_prob_train, na_whole_trips=na_whole_trips, chunk_size=chunk_size)
            df_train_meta = generate_missing_values(df_train_meta, exist_prob=exist_prob_train, na_whole_trips=na_whole_trips, chunk_size=chunk_size)
            df_test = generate_missing_values(df_test, exist_prob=exist_prob_test, na_whole_trips=na_whole_trips, chunk_size=chunk_size)

        # set tags to invalid
        df_train = set_tags_to_invalid(df_train, consider_accel, consider_gyro, consider_gps)
        df_train_meta = set_tags_to_invalid(df_train_meta, consider_accel, consider_gyro, consider_gps)
        df_test = set_tags_to_invalid(df_test, consider_accel, consider_gyro, consider_gps)

        # train model
        if weighted:
            model_dict = train_models_weighted(df_train, df_train_meta, target_name)
        else:
            model_dict = train_models(df_train, df_train_meta, target_name)

        # make predictions on hold out set
        df_y = make_predictions(model_dict, df_test, target_name)

        # get avg acceleration for calculating R^2_oos
        y_train_avg = df_train[target_name].mean()

        # get metrics results
        df_res = get_results(df_y, df_test, target_name, y_train_avg, meta_only=False)
        df_res = df_res.add_suffix(f'_fold_{i + 1}')
        results = pd.concat([results, df_res], axis=1)

    return results


def train_models(df_train, df_train_meta, target_name, return_x_meta=False):
    """
    Trains base models on individual sensor data ('gps' and 'gyro') and a meta model on
    the predictions of the base models combined with raw accelerometer data.

    Parameters:
    - df_train (pd.DataFrame): The training dataset for base models.
    - df_train_meta (pd.DataFrame): The training dataset for the meta model.
    - target_name (str): The target column name to predict.
    - return_x_meta (bool, optional): If True, returns the feature set for the meta model. Defaults to False.

    Returns:
    - dict: A dictionary containing trained base and meta models.
      Keys are: 'gps_model', 'gyro_model', and 'meta_model'.
    - pd.DataFrame (optional): The feature set for the meta model. Returned only if `return_x_meta` is True.

    Notes:
    - The base models are trained using the LightGBM regressor on individual sensor types.
    - The meta model is trained on the predictions made by the base models and the raw phone's accelerometer data.
    - The function facilitates a stacked model training approach.

    Example Usage:
    trained_models = train_models(df_train, df_train_meta, 'target_forward')
    trained_models['gps_model']  # Accessing the trained GPS model.
    """

    # prepare training data
    X_train_gps, X_train_gyro, y_train = create_data_sets(df_train, target_name)
    X_train_gps_meta, X_train_gyro_meta, y_train_meta = create_data_sets(df_train_meta, target_name)
    X_train_dict = {'gps_base': X_train_gps,
                    'gyro_base': X_train_gyro,
                    'gps_meta': X_train_gps_meta,
                    'gyro_meta': X_train_gyro_meta
                    }
    # specify weights for training
    weights_base = 1
    weights_meta = 1

    # accel prediction
    accel_name = 'accel_' + target_name.split('_')[1]
    y_pred_train_meta_accel = np.array(
        df_train_meta[accel_name].fillna(0))  # take the raw phone accel sensor values

    y_pred_dict = {'accel_pred': y_pred_train_meta_accel}

    models_dict = {}

    # train model
    for key in ['gyro', 'gps']:
        # train model
        model = LGBMRegressor(boosting_type='gbdt', objective='regression')
        model.fit(X_train_dict[key + '_base'], y_train, weights_base)

        # store model
        model_key = key + '_model'
        models_dict[model_key] = model

        # make predictions for meta model
        y_pred_train_meta = model.predict(X_train_dict[f"{key}_meta"])

        # store predictions
        pred_key = key + '_model_pred'
        y_pred_dict[pred_key] = y_pred_train_meta

    # create meta training set
    X_train_meta = create_X_meta(y_pred_dict, df_train_meta)

    # fit meta model
    model_meta = LGBMRegressor(boosting_type='gbdt', objective='regression')
    model_meta.fit(X_train_meta, y_train_meta, weights_meta)

    models_dict['meta_model'] = model_meta

    if return_x_meta:
        return models_dict, X_train_meta
    else:
        return models_dict


def train_models_weighted(df_train, df_train_meta, target_name, return_x_meta=False):
    """
    Trains base models + weighted base models on individual sensor data ('gps' and 'gyro') and a meta model
    using predictions of these models combined with raw accelerometer data.

    Parameters:
    - df_train (pd.DataFrame): The training dataset for the base models.
    - df_train_meta (pd.DataFrame): The training dataset for the meta model.
    - target_name (str): The target column name to predict.
    - return_x_meta (bool, optional): If True, returns the feature set for the meta model. Defaults to False.

    Returns:
    - dict: A dictionary containing trained base and meta models.
     The keys include models for 'gps', 'gyro', 'gps_weighted', 'gyro_weighted', and 'meta_model'.
    - pd.DataFrame (optional): The feature set for the meta model. Returned only if `return_x_meta` is True.

    Notes:
    - The base models are trained using the LightGBM regressor on individual sensor types.
     Models with a '_weighted' suffix in the name use the absolute value of the target variable as weights.
    - The meta model is trained on predictions made by the base models and the raw phone's accelerometer data.
    - The function extends the stacked model training approach by considering different weight schemes for training.

    Example Usage:
    trained_models = train_models_weighted(df_train, df_train_meta, 'target_accel')
    trained_models['gyro_weighted_model']  # Accessing the trained gyro model with weights.
    """

    # prepare training data
    X_train_gps, X_train_gyro, y_train = create_data_sets(df_train, target_name)
    X_train_gps_meta, X_train_gyro_meta, y_train_meta = create_data_sets(df_train_meta, target_name)
    X_train_dict = {'gps_base': X_train_gps,
                    'gyro_base': X_train_gyro,
                    'gps_meta': X_train_gps_meta,
                    'gyro_meta': X_train_gyro_meta
                    }

    # accel prediction
    accel_name = 'accel_' + target_name.split('_')[1]
    y_pred_train_meta_accel = np.array(
        df_train_meta[accel_name].fillna(0))  # take the raw phone accel sensor values

    y_pred_dict = {'accel_pred': y_pred_train_meta_accel}

    models_dict = {}

    # train model
    for key in ['gyro', 'gps', 'gps_weighted', 'gyro_weighted']:

        # extract first name
        sensor_name = key.split('_')[0]

        # specify weights for training
        if 'weighted' in key:
            weights_base = abs(y_train)
        else:
            weights_base = 1

        # convert all features to float for onnx
        #X_train_dict[sensor_name + '_base'] = X_train_dict[sensor_name + '_base'].astype("float32")
        # initialize model
        model = LGBMRegressor(boosting_type='gbdt', objective='regression')
        model.fit(X_train_dict[sensor_name + '_base'], y_train, weights_base)

        # save onnx_files
        #convert_to_onnx(model, key, target_name, len(X_train_dict[sensor_name + '_base'].columns))

        # store model
        model_key = key + '_model'
        models_dict[model_key] = model

        # make predictions for meta model
        y_pred_train_meta = model.predict(X_train_dict[f"{sensor_name}_meta"])

        # store predictions
        pred_key = key + '_model_pred'
        y_pred_dict[pred_key] = y_pred_train_meta

    # create meta training set
    X_train_meta = create_X_meta(y_pred_dict, df_train_meta)
    weights_meta = abs(y_train_meta) + 1
    # fit meta model

    X_train_meta = X_train_meta.astype("float32")
    model_meta = LGBMRegressor(boosting_type='gbdt', objective='regression')
    model_meta.fit(X_train_meta, y_train_meta, weights_meta)
    convert_to_onnx(model_meta, "meta", target_name, len(X_train_meta.columns))
    models_dict['meta_model'] = model_meta

    if return_x_meta:
        return models_dict, X_train_meta
    else:
        return models_dict

def convert_to_onnx(model, key, target_name, number_of_features):
    """
    Convert a trained LightGBM regression model to ONNX format.

    This function takes a trained LightGBM model and converts it into the ONNX format, which is a standard format
    for representing machine learning models. The converted model can be used with different tools and platforms
    that support ONNX, enabling interoperability between different frameworks.

    Parameters:
    - model: The trained LightGBM regression model to be converted.
    - key (str): A key or identifier used for naming the output ONNX file.
    - target_name (str): The name of the target variable used in the model.
    - number_of_features (int): The number of input features expected by the model.

    Returns:
    None

    Notes:
    - The ONNX model is checked for correctness using the ONNX checker.
    - The converted ONNX model is saved to a file in the "onnx_models" directory, named "{key}_{target_name}.onnx".
    """
    initial_type = [('float_input', FloatTensorType([None, number_of_features]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_type, target_opset=8)
    onnx.checker.check_model(onnx_model)
    with open(f"onnx_models/{key}_{target_name}.onnx", "wb") as f:
        f.write( onnx_model.SerializeToString())

def make_predictions(models_dict, df_test, target_name, return_dfs=False):
    """
    Generates predictions using base models and a meta model for the provided test data.

    Parameters:
    - models_dict (dict): Dictionary storing the trained models, including specific sensor models and the 'meta_model'.
    - df_test (pd.DataFrame): The test dataset for predictions.
    - target_name (str): Name of the target column in `df_test`.
    - return_dfs (bool, optional): If True, returns the test sets for all models including the meta model. Defaults to False.

    Returns:
    - pd.DataFrame: A dataframe comprising predictions from all models, the true target values, and additional context columns.
    - dict (optional): Dictionary containing test sets used for each model's predictions. Returned only if `return_dfs` is True.

    Notes:
    - The function first makes predictions using each base model on their respective test data.
    - A meta test set is constructed using the base model predictions.
    - The meta model is then applied to this set to make final predictions.
    - The output dataframe aggregates all predictions, providing a comprehensive view of the model outcomes.
    """

    # Create testing datasets for GPS and Gyro model
    X_test_gps, X_test_gyro, y_test = create_data_sets(df_test, target_name)

    # Store the test sets in a dictionary for easier access
    X_test_dict = {'gps': X_test_gps, 'gyro': X_test_gyro}

    # Get raw acceleration values from the smartphone sensor and fill NA values with 0
    accel_name = 'accel_' + target_name.split('_')[1]
    y_pred_test_accel = np.array(df_test[accel_name].fillna(0))

    # Initialize the dictionary to store all model predictions with the smartphone sensor predictions
    y_pred_dict = {'accel_pred': y_pred_test_accel}

    # Iterate through each base model in the model dictionary
    for model_key in models_dict:
        # Skip the meta model in this loop, we will handle it separately
        if model_key != 'meta_model':
            # Extract sensor name, it's either gps or gyro
            sensor_name = model_key.split('_')[0]

            # Get the relevant model and corresponding test set
            model = models_dict[model_key]
            X_test = X_test_dict[sensor_name]

            # Make predictions for the current model
            y_pred = model.predict(X_test)

            # Store the predictions in the dictionary
            pred_key = model_key + '_pred'
            y_pred_dict[pred_key] = y_pred

    # After making predictions for all base model, handle the meta model

    # Create meta test set using the predictions made by the base model
    X_test_meta = create_X_meta(y_pred_dict, df_test)

    # Get the meta model
    model_meta = models_dict['meta_model']

    # Make predictions for the meta model
    y_pred_test_meta = model_meta.predict(X_test_meta)

    # Create a dataframe to store all the predictions, including the meta model
    df_y = pd.DataFrame(
        data={target_name: df_test[target_name].values.squeeze(), 'meta_model_pred': y_pred_test_meta, **y_pred_dict})

    # Add additional columns 'driveID' and 'timestamp_utc' to the dataframe
    df_y = pd.concat([df_test[['driveID', 'timestamp_utc']].reset_index(drop=True), df_y], axis=1)

    # If the return_dfs flag is True, return the predictions dataframe and the test sets dictionary
    # Otherwise, return only the predictions dataframe
    if return_dfs:
        X_test_dict['meta_model'] = X_test_meta
        return df_y, X_test_dict
    else:
        return df_y


def r2_score_oos(y_true, y_pred, y_train_avg):
    """
     Computes the R-squared for out-of-sample data.

     Parameters:
     - y_true (array-like): True target values.
     - y_pred (array-like): Predicted values.
     - y_train_avg (float): Mean of the target values from the training set.

     Returns:
     - float: The R-squared value. A value of 1 indicates perfect predictions,
       while a value of 0 indicates that the model is no better than a model
       that simply predicts the mean of the target values from the training set.
       Negative values are possible as well, indicating worse performance than
       predicting the mean.
    """
    r2 = 1 - sum(((y_true - y_pred) ** 2) / sum((y_true - y_train_avg) ** 2))
    return r2


def get_results(df_y, df_test, target_name, y_train_avg, meta_only=False):
    """
    Evaluate multiple prediction models on various performance metrics.

    Parameters:
    - df_y (pd.DataFrame): DataFrame containing true target values and predicted values from multiple models.
    - df_test (pd.DataFrame): Test dataset with additional columns (e.g. 'gps_speed') required for evaluation.
    - target_name (str): Name of the target column in the df_y dataframe.
    - y_train_avg (float): Mean of the target values from the training dataset.
    - meta_only (bool, optional): If True, evaluates only the meta model. Defaults to False.

    Returns:
    - pd.DataFrame: A results dataframe where rows represent metrics and columns represent models.
      Each cell contains the metric score for the respective model.
    """
    if meta_only:
        models = ['meta_model_pred']
    else:
        models = df_y.columns[df_y.columns.str.contains('pred')]

    y_test = df_y.loc[:, target_name]
    y_preds = df_y.loc[:, models]

    df_results = pd.DataFrame()

    df_2 = df_y[abs(df_y[target_name]) > 2].copy()
    df_3 = df_y[abs(df_y[target_name]) > 3].copy()

    threshold_kmh = 10
    mask = np.array(df_test['gps_speed'] > (threshold_kmh / 3.6))
    df_4 = df_y[mask].copy()
    for model in models:
        mse = metrics.mean_squared_error(y_test, y_preds[model])
        r2 = r2_score_oos(y_test, y_preds[model], y_train_avg)
        mae = metrics.mean_absolute_error(y_test, y_preds[model])
        mse2 = metrics.mean_squared_error(df_2[target_name], df_2[model])
        mse3 = metrics.mean_squared_error(df_3[target_name], df_3[model])
        mse10 = metrics.mean_squared_error(df_4[target_name], df_4[model])
        mae2 = metrics.mean_absolute_error(df_2[target_name], df_2[model])
        mae3 = metrics.mean_absolute_error(df_3[target_name], df_3[model])
        mae10 = metrics.mean_absolute_error(df_4[target_name], df_4[model])
        # Fehlermaß auf Basis der maximumsnorm
        max_norm = df_y.groupby('driveID').apply(lambda x: abs(x[target_name] - x[model]).max()).mean()

        df_results.loc['r2', model] = np.round(r2, 3)
        df_results.loc['mae', model] = np.round(mae, 4)
        df_results.loc['mse', model] = np.round(mse, 3)
        df_results.loc['mae>2', model] = np.round(mae2, 3)
        df_results.loc['mse>2', model] = np.round(mse2, 3)
        df_results.loc['mae>3', model] = np.round(mae3, 3)
        df_results.loc['mse>3', model] = np.round(mse3, 3)
        df_results.loc['mae>10km/h', model] = np.round(mae10, 3)
        df_results.loc['mse>10km/h', model] = np.round(mse10, 3)
        df_results.loc['max_absolute_diff', model] = np.round(max_norm, 3)

    return df_results


def custom_shap_bar_plot(shap_values, feature_names, max_display=10, fig_size=(10, 7), color='crimson', filesave=None):
    """
    Visualize the magnitude of SHAP values for model features in a custom bar plot.
    It gives more control over certain aspects of the visualization compared to the
    standard bar plot from the SHAP python package.

    Parameters:
    - shap_values (np.array or pd.DataFrame): Array or DataFrame of SHAP values for each feature.
    - feature_names (list or pd.Index): List of feature names corresponding to the SHAP values.
    - max_display (int, optional): Maximum number of top features to display. Defaults to 10.
    - fig_size (tuple, optional): Size of the figure in inches (width, height). Defaults to (10, 7).
    - color (str, optional): Color for the bars. Defaults to 'crimson'.
    - filesave (str, optional): Path to save the plot. If None, the plot is not saved. Defaults to None.

    Returns:
    - None: Displays the bar plot.

    Notes:
    - The function calculates the mean absolute SHAP value for each feature and then sorts them.
    - If the total number of features exceeds the `max_display`, the function will aggregate the remaining features into a single category.
    - Features with higher SHAP values have higher model impact.
    """

    # berechne durchschnittliche shap values und sortiere sie absteigend
    mean_shap = pd.DataFrame(abs(pd.DataFrame(shap_values, columns=feature_names)).mean(axis=0))
    mean_shap = mean_shap.sort_values(ascending=True, by=mean_shap.columns[0])

    best_feat = mean_shap.iloc[-max_display:]

    # falls nicht alle Features gezeigt werden können, summiere diese auf und zeige sie in der letzten Spalte
    if max_display < len(feature_names):
        sum_others = mean_shap.iloc[:(1 - max_display)].sum()
        no_of_features = len(mean_shap) + (1 - max_display)
        message = str('sum of ') + str(no_of_features) + str(' other features')
        best_feat = best_feat.rename(index={best_feat.index[0]: message})
        best_feat.at[best_feat.index[0], best_feat.columns[0]] = sum_others

    # wandle df in Series
    val = best_feat.squeeze()

    fig, ax = plt.subplots(figsize=fig_size)

    bars = ax.barh(val.index, val, height=0.7, color=color)

    ax.yaxis.set_ticks_position('none')

    ax.set_xlabel('mean (|SHAP Value|)', fontsize=13)
    ax.tick_params(axis='y', labelsize=13)

    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dotted', linewidth=0.3)

    ticklabel = [f"+{a:.3f}" for a in val]

    ax.bar_label(bars, padding=5, labels=ticklabel, color='black', fontsize=13)

    if filesave is not None:
        plt.savefig(filesave, bbox_inches='tight')
    plt.show()


def rgb_to_str(rgb):
    """Convert RGB tuple with values between 0 and 1 to a string format."""
    return 'rgb({:.0f}, {:.0f}, {:.0f})'.format(*(x * 255 for x in rgb))


def calculate_metric(df, target_name, metric_func, model):
    """
    Calculate a specified metric for each unique 'driveID' in the dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe containing data for multiple 'driveID's.
    - target_name (str): Name of the target column in the dataframe.
    - metric_func (function): A function that computes the desired metric.
                              It should accept two arguments: actual values and predicted values.
    - model (str): Name of the column in the dataframe containing the predicted values.

    Returns:
    - pd.Series: A series containing the computed metric for each unique 'driveID'.

    Notes:
    - This function is designed to compute metrics (e.g., MSE, MAE) over segments of data grouped by 'driveID'.
    """
    return df.groupby('driveID').apply(lambda x: metric_func(x[target_name], x[model]))


def plot_driveIDs2(df_y, target_name, model_list=None, d_id_list=None, rows=2, cols=2, figsize=(16, 12),
                   metric='mse',
                   typ='best', random_state=42, save=False, path=None, metric_model='meta_model_pred'):
    """
    Plot actual vs. predicted values for specified driveIDs using Plotly.

    Parameters:
    - df_y (pd.DataFrame): DataFrame containing the target and prediction values.
    - target_name (str): Name of the column containing the actual target values.
    - model_list (list, optional): List of model names to be plotted. Default is target_name and meta_model_pred.
    - d_id_list (list, optional): List of driveIDs to plot. Overrides the 'typ' parameter if provided.
    - rows (int): Number of rows in the subplot grid.
    - cols (int): Number of columns in the subplot grid.
    - figsize (tuple): Size of the entire plot.
    - metric (str, optional): Metric by which to sort and select driveIDs. Default is 'mse'.
    - typ (str): Type of driveIDs to select: 'best', 'worst', 'random', 'longest', or 'shortest'.
    - random_state (int): Seed for random number generation. Used when typ='random'.
    - save (bool): Whether to save the plot to a file.
    - path (str, optional): Path to save the plot, if save=True.
    - metric_model (str, optional): Model name used to calculate the metric. Default is 'meta_model_pred'.

    Returns:
    - plotly.graph_objs._figure.Figure: A Plotly figure object with subplots.

    Notes:
    - This function is designed to visualize and compare the actual vs. predicted values for
      different models across specified driveIDs using Plotly.
    - The 'typ' parameter determines the criteria for selecting driveIDs. If a list of driveIDs is
      provided through 'd_id_list', it will override the 'typ' criteria.
    """

    metric_funcs = {
        'mse': metrics.mean_squared_error,
        'mape': metrics.mean_absolute_percentage_error,
        'mae': metrics.mean_absolute_error,
        'r2': metrics.r2_score
    }

    # legend on the bottom
    models_legend_naming = {'target_left': 'Telematics Sensor Left',
                            'target_forward': 'Telematics Sensor Forward',
                            'accel_pred': 'Phone Acceleration',
                            'gps_model_pred': 'Phone GPS Model',
                            'gyro_model_pred': 'Phone Gyro Model',
                            'gps_weighted_model_pred': 'Phone GPS Model weighted',
                            'gyro_weighted_model_pred': 'Phone Gyro Model weighted',
                            'meta_model_pred': 'Meta Model',
                            }

    size = rows * cols

    if typ == 'random':
        unique_dIDs = df_y['driveID'].unique()
        if len(unique_dIDs) > size:
            ordered_ids = np.random.RandomState(random_state).choice(unique_dIDs, size=size, replace=False)
        else:
            ordered_ids = unique_dIDs

    if d_id_list is not None:
        ordered_ids = d_id_list
        size = len(ordered_ids)
        rows = 1
        cols = 1

    if typ in ['best', 'worst']:
        driveIDs = calculate_metric(df_y, target_name, metric_funcs[metric], metric_model)
        if typ == 'worst':
            if metric == 'r2':
                ordered_ids = driveIDs.sort_values(
                    ascending=True).index  # For 'r2', 'worst' means the smallest values
            else:
                ordered_ids = driveIDs.sort_values(ascending=False).index
        else:  # typ == 'best'
            if metric == 'r2':
                ordered_ids = driveIDs.sort_values(
                    ascending=False).index  # For 'r2', 'best' means the largest values
            else:
                ordered_ids = driveIDs.sort_values(ascending=True).index
    elif typ == 'longest':
        drive_counts = df_y['driveID'].value_counts().sort_values(ascending=False)
        ordered_ids = drive_counts[:size].index
    elif typ == 'shortest':
        drive_counts = df_y['driveID'].value_counts().sort_values(ascending=True)
        ordered_ids = drive_counts[:size].index

    if model_list is None:
        model_list = [target_name, 'meta_model_pred']

    # truncate ordered_ids
    ordered_ids = ordered_ids[: rows * cols]

    colors = {}
    for i, model in enumerate(model_list):
        colors[i] = rgb_to_str(sns.color_palette("deep", len(model_list))[i])

    def annotate_plot(fig, row, col, df, d_id, model_list, colors, metric, idx_subplot):
        df_sub = df[df['driveID'] == np.int64(d_id)]

        for j, model in enumerate(model_list):
            fig.add_trace(go.Scatter(x=df_sub['timestamp_utc'],
                                     y=df_sub[model],
                                     line=dict(color=colors[j], width=3),
                                     name=models_legend_naming[model]),
                          row=row, col=col)

        # Update y and x axis for the plot
        fig.update_yaxes(title_text='acceleration [' + r'$m/s^2$' + ']',
                         showgrid=True,
                         showline=True,
                         row=row,
                         col=col)

        fig.update_xaxes(title_text='time [' + r'$hh:mm:ss$' + ']',
                         tickangle=30,
                         showgrid=True,
                         showline=True,
                         row=row,
                         col=col)

        # Add title as an annotation
        metric_score = calculate_metric(df_sub, target_name, metric_funcs[metric], metric_model).values[0]
        title = f'DriveID: {d_id} – {metric}: {metric_score:.2f}'
        fig.layout.annotations[idx_subplot]['text'] = title

    assert len(ordered_ids) <= rows * cols, f"Too many IDs: {len(ordered_ids)} vs. Grid: {rows * cols}"

    # Create a subplot grid with temporary subtitles
    fig = make_subplots(rows=rows, cols=cols, shared_yaxes=True, subplot_titles=['temp_subtitle' for i in ordered_ids])

    for i, d_id in enumerate(ordered_ids):
        row = i // cols + 1
        col = i % cols + 1
        annotate_plot(fig, row, col, df_y, d_id, model_list, colors, metric, i)

    # Convert to pixels using 96 DPI
    dpi = 96
    figsize_pixels = (figsize[0] * dpi, figsize[1] * dpi)

    fig.update_layout(legend=dict(yanchor="middle", y=0.5, xanchor="center", x=1.2),
                      margin=dict(b=150),
                      width=figsize_pixels[0],
                      height=figsize_pixels[1])

    if save:
        plt.savefig(path, bbox_inches='tight')

    return fig


# store dataframe for streamlit
def export_to_streamlit(df_y, df_test, save_path=None):
    """
   Serialize and save the results dataframe for visualization in Streamlit.

   Parameters:
   - df_y (pd.DataFrame): DataFrame containing true target values and predicted values.
   - df_test (pd.DataFrame): Test dataset with additional columns required for visualization.
   - save_path (str, optional): Designated path to save the serialized dataframe.

   Notes:
   - The function first resets the indices of the input dataframes.
   - If a file already exists at the specified path, the function modifies the filename to prevent overwriting.
   - Only specific columns from the `df_test` are retained in the final dataframe.
   """

    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    if save_path is None:
        save_path = os.path.join(path, '/results/streamlit_df.pkl')

    df_y.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # Check if the file exists and modify the filename if necessary
    base_path, ext = os.path.splitext(save_path)
    counter = 1
    while os.path.exists(save_path):
        save_path = f"{base_path}_{counter}{ext}"
        counter += 1

    status = list(df_test.columns[df_test.columns.str.contains('valid')])
    status.append('gps_speed')
    exp_df = pd.concat([df_y, df_test[status]], axis=1)
    exp_df.to_pickle(save_path)

    print(f'file saved to {save_path}')








# ähnliche Funktionalität wie plot_driveIDs2 aber mit matplotlib statt plotly

def plot_driveIDs(df_y, target_name, model_list=None, d_id_list=None, rows=2, cols=2, figsize=(16, 12), metric='mse',
                  typ='best', random_state=42, save=False, path=None, metric_model='meta_model_pred'):
    # type : best, worse, random

    metric_funcs = {
        'mse': metrics.mean_squared_error,
        'mape': metrics.mean_absolute_percentage_error,
        'mae': metrics.mean_absolute_error,
        'r2': metrics.r2_score
    }

    # legend on the bottom
    models_legend_naming = {'target_left': 'Telematics Sensor Left',
                            'target_forward': 'Telematics Sensor Forward',
                            'accel_pred': 'Phone Acceleration',
                            'gps_model_pred': 'Phone GPS Model',
                            'gyro_model_pred': 'Phone Gyro Model',
                            'gps_weighted_model_pred': 'Phone GPS Model weighted',
                            'gyro_weighted_model_pred': 'Phone Gyro Model weighted',
                            'meta_model_pred': 'Meta Model',
                            }

    def calculate_metric(df, target_name, metric_func, model=metric_model):
        return df.groupby('driveID').apply(lambda x: metric_func(x[target_name], x[model]))

    size = rows * cols

    if typ == 'random':
        unique_dIDs = df_y['driveID'].unique()
        if len(unique_dIDs) > size:
            ordered_ids = np.random.RandomState(random_state).choice(unique_dIDs, size=size, replace=False)
        else:
            ordered_ids = unique_dIDs

    if d_id_list is not None:
        ordered_ids = d_id_list
        size = len(ordered_ids)
        rows = 1
        cols = 1

    if typ in ['best', 'worst']:
        driveIDs = calculate_metric(df_y, target_name, metric_funcs[metric])
        if typ == 'worst':
            if metric == 'r2':
                ordered_ids = driveIDs.sort_values(ascending=True).index  # For 'r2', 'worst' means the smallest values
            else:
                ordered_ids = driveIDs.sort_values(ascending=False).index
        else:  # typ == 'best'
            if metric == 'r2':
                ordered_ids = driveIDs.sort_values(ascending=False).index  # For 'r2', 'best' means the largest values
            else:
                ordered_ids = driveIDs.sort_values(ascending=True).index
    elif typ == 'longest':
        drive_counts = df_y['driveID'].value_counts().sort_values(ascending=False)
        ordered_ids = drive_counts[:size].index
    elif typ == 'shortest':
        drive_counts = df_y['driveID'].value_counts().sort_values(ascending=True)
        ordered_ids = drive_counts[:size].index

    if model_list is None:
        model_list = [target_name, 'meta_model_pred']

    fig, axs = plt.subplots(rows, cols, figsize=figsize, sharey='all')  # Create a 2x2 subplot grid
    fig.text(0.5, -0.02, 'time [' + r'$hh:mm:ss$' + ']', ha='center', size=20)  # Set the x-axis label for all subplots

    fig.text(-0.02, 0.5, 'acceleration [' + r'$m/s^2$' + ']', va='center', rotation='vertical', size=20)

    # ordne colors zu für die einzelnen kategorien
    colors = {}
    for i, model in enumerate(model_list):
        colors[i] = sns.color_palette("deep", len(model_list))[i]

    def annotate_plot(ax, df, d_id, model_list, colors, metric):
        df_sub = df[df['driveID'] == d_id]
        for j, model in enumerate(model_list):
            ax.plot(df_sub['timestamp_utc'], df_sub[model], linewidth=3, color=colors[j])
        for label in ax.get_xticklabels():
            label.set_rotation(30)  # Rotate x-axis tick labels by 30 degrees
            label.set_fontsize(12)  # Set the font size to 12

        metric_score = calculate_metric(df_sub, target_name, metric_funcs[metric]).values[0]
        # ax.text(0.95, 0.95, f'{metric}: {metric_score:.2f}', transform=ax.transAxes,
        #        verticalalignment='top', horizontalalignment='right')
        ax.set_title(f'DriveID: {d_id} – {metric}: {metric_score:.2f}', fontsize=20)

        # models_legend = [models_legend_naming[model] for model in model_list]
        # ax.legend(models_legend)

        ax.grid(True)

    for ax, d_id in zip(axs.flatten(), ordered_ids):
        annotate_plot(ax, df_y, d_id, model_list, colors, metric)

    models_legend = [models_legend_naming[model] for model in model_list]
    fig.legend(models_legend, loc='lower center', bbox_to_anchor=(0.5, -0.2), prop={'size': 20})

    # Adjust the layout to include the legend
    fig.tight_layout(rect=(0, 0, 1, 1))
    plt.tight_layout()

    if save:
        plt.savefig(path, bbox_inches='tight')
    # plt.show()
    return fig

