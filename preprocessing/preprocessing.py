import sqlite3
from fastai.tabular.core import add_datepart
import os
import sys

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PROJECT_PATH)

from preprocessing.preprocessing_utils import *

def dataloader_1hz(filepath="/home/stud03/data_science_challenge/data/kit_telematik_dats.sqlite", group_id=1,
                   get_gyro=True, get_accel=True, get_timestamps=True, get_cyclical_time=True,
                   get_weather=True, get_surrounding_changes=True, number_of_timesteps=5, train_split=0.8,
                   shuffle_split=True, min_entry_driveid=15, invalid_tags_to_nan = False, drop_nans=False,
                   max_amount=None, save=True, savedir=None, savename=None, verbose=False, get_coord_accel=True):

    """
    Load (Phone GPS and Telematics Sensor, other phone sensors also included if flags are set accordingly) data corresponding to a specific group_id from a SQL database,
    transform it to 1Hz and perform further preprocessing steps as indicated by the parameters.

    Parameters:
    - filepath (string): The filepath of the SQL database.
    - group_id (int): The wanted group_id (must be between [1, ..., 9]).
    - get_gyro (bool): Wether data from the phone gyro sensor should be included.
    - get_accel (bool): Wether data from the phone acceleration sensor should be included.
    - get_timestamps (bool): Wether to calculate diverse timestamp-information features from the given timestamps.
    - get_cyclical_time (bool): Wether to also compute cyclical timestamp-features from the features given above.
    - get_weather (bool): Wether to get weather data at the given GPS coordinates and timestamps.
    - get_surrounding_changes (bool): Wether to include the features from the <T> surrounding 1Hz-timestamps into each timestamp.
    - number_of_timestamps (int): <T> feeding into the above parameter.
    - train_split (float): Ratio of train-test-split.
    - shuffle_split (bool): Wether to randomly shuffle the driueIDs before assigning them to either train or test.
    - min_entry_driveid (int): The minimum amount of 1Hz datapoints a driveID must have in order to be included in the output.
    - invalid_tags_to_nan (bool): Wether to transform measurements which are tagged as invalid to NaN.
    - drop_nans (bool): Wether to drop rows with nan values.
    - max_amount (int): The max amount of 1Hz datapoints to be returned.
    - save (bool): Wether to save the resulting dataframe.
    - savedir (string): The directory to save it in.
    - savename (string): The name to save it as.
    - verbose (bool): Wether to print information about the preprocessing progress in the command line.
    - get_coord_accel (bool): Wether to also calculate the gps-based acceleration as the second derivative of coordinates and not only as the first derivative of speed.

    Returns:
    - df_train (dataframe): DataFrame consisting of 1Hz measurements and calculated features used for training.
    - df_test (dataframe): DataFrame consisting of 1Hz measurements and calculated features used for testing.

    Notes:
    - Our database consists of 9 driving situations, so that is why the group_id must between 1 and 9.
    - train_split = 1.0 results in an empty 'test'-df thus enabling crossvalidation on the train-df.
    """

    tqdm.pandas()
    # Startup

    con = sqlite3.connect(filepath)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    cur.fetchall()

    # Phone GPS data
    if verbose:
        print("Reading GPS Data")
    if max_amount is None:
        df_features = pd.read_sql_query(
            f"SELECT driveID, timestamp_utc, groupID, osmID, latitude, longitude, direction, speed, gps_status FROM position WHERE groupID = %s" % (
                group_id), con)
    else:
        driveIDs = pd.read_sql_query(
            f"SELECT distinct driveID FROM position WHERE groupID = %s limit %s" % (group_id, max_amount), con)
        list_driveIDs = list(driveIDs['driveID'])
        placeholder = '?'  # For SQLite. See DBAPI paramstyle.
        placeholders = ', '.join(placeholder for unused in list_driveIDs)
        query = 'SELECT driveID, timestamp_utc, groupID, osmID, latitude, longitude, direction, speed, gps_status FROM position WHERE driveID IN (%s)' % placeholders
        df_features = pd.read_sql_query(query, con, params=list_driveIDs)
    df_features = df_features.rename(
        columns={'latitude': 'gps_lat', 'longitude': 'gps_long', 'direction': 'gps_direction', 'speed': 'gps_speed',
                 'gps_status': 'gps_valid'})
    df_features['gps_valid'] = df_features['gps_valid'].map({'NOT_VALID': 0, 'VALID': 1, 'DATA_MISSING': 0})
    df_features['gps_direction'] = df_features['gps_direction'].map({'FORWARD': 1, 'BACKWARD': -1, 'DATA_MISSING': 0})
    df_features = df_features.astype(
        {'groupID': 'int32', 'osmID': 'int32', 'gps_lat': 'float32', 'gps_long': 'float32', 'gps_speed': 'float32',
         'gps_valid': 'int32', 'gps_direction': 'int32'})
    pd.set_option('mode.chained_assignment', None)
    df_features["timestamp_utc"] = [x[:-4] for x in df_features["timestamp_utc"]]
    df_features["timestamp_utc"] = pd.to_datetime(df_features["timestamp_utc"])
    # Calculate Acceleration Based on GPS Data
    if verbose:
        print("Calculating Acceleration from GPS Data")
    df_features = df_features.groupby('driveID').progress_apply(lambda x: calculate_acceleration(x))
    df_features = df_features.reset_index(drop=True)
    # Calculate GPS Mercator Difference between prev and current timestep
    if verbose:
        print("Calculating GPS Change")
    df_features["gps_lat_m"], df_features["gps_long_m"] = convert_coordinates_to_mercator(df_features["gps_lat"], df_features["gps_long"])
    df_features = apply_coordinates_calc_diff(df_features)
    df_features = df_features.reset_index(drop=True)
    if get_coord_accel:
        if verbose:
            print('Calculating speed and accel from gps coords')
        df_features = calculate_accel_from_gps(df_features)
        df_features = df_features.reset_index(drop=True)

    # Phone Gyro Sensor Data
    if get_gyro:
        if verbose:
            print("Reading Gyro Data")
        if max_amount is None:
            df_gyro = pd.read_sql_query(
                f"SELECT driveID, timestamp_utc, groupID, roll, pitch, yaw, phone_gyro_status FROM phone_gyro WHERE groupID = %s" % (
                    group_id), con)
        else:
            query = 'SELECT driveID, timestamp_utc, groupID, roll, pitch, yaw, phone_gyro_status FROM phone_gyro WHERE driveID IN (%s)' % placeholders
            df_gyro = pd.read_sql_query(query, con, params=list_driveIDs)
        df_gyro = df_gyro.rename(
            columns={'roll': 'gyro_roll', 'pitch': 'gyro_pitch', 'yaw': 'gyro_yaw', 'phone_gyro_status': 'gyro_valid'})
        df_gyro['gyro_valid'] = df_gyro['gyro_valid'].map({'NOT_VALID': 0, 'VALID': 1, 'DATA_MISSING': 0})
        df_gyro = df_gyro.astype(
            {'groupID': 'int32', 'gyro_roll': 'float32', 'gyro_yaw': 'float32', 'gyro_pitch': 'float32',
             'gyro_valid': 'int32'})
        df_gyro['timestamp_utc'] = df_gyro['timestamp_utc'].apply(lambda x: x[:-4])
        # Sample to 1Hz
        df_gyro = df_gyro.groupby("timestamp_utc", as_index=False).agg({
            'driveID': 'first',
            'groupID': 'first',
            'gyro_roll': 'mean',
            'gyro_pitch': 'mean',
            'gyro_yaw': 'mean',
            'gyro_valid': all_valid
        })
        df_gyro = df_gyro.rename(columns={'driveID first': 'driveID', 'groupID first': 'groupID',
                                          'gyro_valid first': 'gyro_valid',
                                          'gyro_valid all_values': 'gyro_valid'})
        df_gyro.columns = df_gyro.columns.str.replace(' ', '_')
        df_gyro["timestamp_utc"] = pd.to_datetime(df_gyro["timestamp_utc"])
        if verbose:
            print('done')
        # Merge to features
        df_features = pd.merge(df_features, df_gyro, on=['driveID', 'timestamp_utc', 'groupID'])

        del df_gyro

    # Phone Acceleration Sensor Data
    if get_accel:
        if verbose:
            print("Reading Accel Data")
        if max_amount is None:
            df_accel = pd.read_sql_query(
                f"SELECT driveID, timestamp_utc, groupID, forward, left, up, phone_accel_status FROM phone_accel WHERE groupID = %s" % (
                    group_id), con)
        else:
            query = 'SELECT driveID, timestamp_utc, groupID, forward, left, up, phone_accel_status FROM phone_accel WHERE driveID IN (%s)' % placeholders
            df_accel = pd.read_sql_query(query, con, params=list_driveIDs)
        df_accel = df_accel.rename(columns={'forward': 'accel_forward', 'left': 'accel_left', 'up': 'accel_up',
                                            'phone_accel_status': 'accel_valid'})
        df_accel['accel_valid'] = df_accel['accel_valid'].map({'NOT_VALID': 0, 'VALID': 1, 'DATA_MISSING': 0})
        df_accel = df_accel.astype(
            {'groupID': 'int32', 'accel_forward': 'float32', 'accel_left': 'float32', 'accel_up': 'float32',
             'accel_valid': 'int32'})
        df_accel["timestamp_utc"] = [x[:-4] for x in df_accel["timestamp_utc"]]
        # Sample to 1 Hz
        df_accel = df_accel.groupby("timestamp_utc", as_index=False).agg({
            'driveID': 'first',
            'groupID': 'first',
            'accel_forward': 'mean',
            'accel_left': 'mean',
            'accel_up': 'mean',
            'accel_valid': all_valid
        })
        df_accel = df_accel.rename(columns={'driveID first': 'driveID', 'groupID first': 'groupID',
                                            'accel_valid first': 'accel_valid'})
        df_accel.columns = df_accel.columns.str.replace(' ', '_')
        df_accel["timestamp_utc"] = pd.to_datetime(df_accel["timestamp_utc"])
        # Merge to features
        df_features = pd.merge(df_features, df_accel, on=['driveID', 'timestamp_utc', 'groupID'])

        del df_accel

    # Read target: Car acceleration sensor
    if verbose:
        print("Reading Target Data")
    if max_amount is None:
        df_target = pd.read_sql_query(f"SELECT driveId, timestamp_utc, groupID, forward, left, up, tag_accel_status FROM tag_accel WHERE groupID = %s" % (group_id), con)
    else:
        query = 'SELECT driveID, timestamp_utc, groupID, forward, left, up, tag_accel_status FROM tag_accel WHERE driveID IN (%s)' % placeholders
        df_target = pd.read_sql_query(query, con, params=list_driveIDs)
    df_target = df_target.rename(columns={'forward': 'target_forward', 'left': 'target_left', 'up': 'target_up',
                                          'tag_accel_status': 'target_valid'})
    df_target['target_valid'] = df_target['target_valid'].map({'NOT_VALID': 0, 'VALID': 1, 'DATA_MISSING': 0})
    df_target = df_target.astype(
        {'groupID': 'int32', 'target_forward': 'float32', 'target_left': 'float32', 'target_up': 'float32',
         'target_valid': 'int32'})
    df_target["timestamp_utc"] = [x[:-4] for x in df_target["timestamp_utc"]]
    # Sample to 1 Hz
    df_target = df_target.groupby("timestamp_utc", as_index=False).agg({
        'driveID': 'first',
        'groupID': 'first',
        'target_forward': 'mean',
        'target_left': 'mean',
        'target_up': 'mean',
        'target_valid': all_valid
    })
    df_target = df_target.rename(
        columns={'driveID first': 'driveID', 'groupID first': 'groupID', 'target_valid all_valid': 'target_valid'})
    df_target.columns = df_target.columns.str.replace(' ', '_')
    df_target["timestamp_utc"] = pd.to_datetime(df_target["timestamp_utc"])

    # Close SQLite Connection
    con.close()



    # Merge
    if verbose:
        print("Merging Data")
    df_merged = pd.merge(df_features, df_target, on=['driveID', 'timestamp_utc', 'groupID'])

    # Convert valid tags to bools
    column_types = {
        'accel_valid': bool,
        'gps_valid': bool,
        'target_valid': bool,
        'gyro_valid': bool, }

    df_merged = df_merged.astype(column_types)

    # Delete driveID's where target is invalid
    if verbose:
        print("Removing Corrupt Data")
    corrupt_drive_ids = list(set(df_target[df_target['target_valid'] == False]['driveID']))
    df_merged = df_merged[~df_merged['driveID'].isin(corrupt_drive_ids)]

    # Delete driveIDs with not enough data
    if verbose:
        print("Removing Short Trips")
    if min_entry_driveid > 0:
        df_merged['ID_count'] = df_merged.groupby('driveID')['driveID'].transform('count')
        # Filter rows based on the count condition
        df_merged = df_merged[df_merged['ID_count'] >= min_entry_driveid].copy()
        # Remove the 'ID_count' column
        df_merged.drop('ID_count', axis=1, inplace=True)

    # Index driveID
    if verbose:
        print("Indexing DriveID")
    distinct_ids = df_merged['driveID'].unique()
    id_mapping = {id_val: idx for idx, id_val in enumerate(distinct_ids)}
    df_merged['driveID'] = df_merged['driveID'].map(id_mapping)
    df_merged = df_merged.astype({'driveID': 'int32'})

    # Transform timestamps
    if get_timestamps:
        if verbose:
            print("Transforming Timestamps")
        df_merged = add_datepart(df_merged, 'timestamp_utc', time=True, drop=False)
        df_merged.drop(['timestamp_utcIs_month_end', 'timestamp_utcIs_month_start', 'timestamp_utcYear',
                        'timestamp_utcIs_quarter_end', 'timestamp_utcIs_quarter_start', 'timestamp_utcIs_year_end',
                        'timestamp_utcIs_year_start', 'timestamp_utcElapsed', 'timestamp_utcSecond',
                        'timestamp_utcMinute'], axis=1, inplace=True)
        df_merged = df_merged.rename(
            columns={'timestamp_utcMonth': 'timestamp_month', 'timestamp_utcWeek': 'timestamp_week',
                     'timestamp_utcDayofweek': 'timestamp_dayofweek', 'timestamp_utcHour': 'timestamp_hour'})

    # Sinus-Cosinus Encodings
    if get_cyclical_time and get_timestamps:
        if verbose:
            print("Creating Cyclical Encoding")
        df_merged['timestamp_cyclical_month_sin'] = np.sin(2 * np.pi * df_merged['timestamp_month'] / 12.0)
        df_merged['timestamp_cyclical_month_cos'] = np.cos(2 * np.pi * df_merged['timestamp_month'] / 12.0)
        df_merged['timestamp_cyclical_day_sin'] = np.sin(2 * np.pi * df_merged['timestamp_week'] / 7.0)
        df_merged['timestamp_cyclical_day_cos'] = np.cos(2 * np.pi * df_merged['timestamp_week'] / 7.0)
        df_merged['timestamp_cyclical_dayofweek_sin'] = np.sin(2 * np.pi * df_merged['timestamp_dayofweek'] / 7.0)
        df_merged['timestamp_cyclical_dayofweek_cos'] = np.cos(2 * np.pi * df_merged['timestamp_dayofweek'] / 7.0)
        df_merged['timestamp_cyclical_hour_sin'] = np.sin(2 * np.pi * df_merged['timestamp_hour'] / 24.0)
        df_merged['timestamp_cyclical_hour_cos'] = np.cos(2 * np.pi * df_merged['timestamp_hour'] / 24.0)
        df_merged = df_merged.astype(
            {'accel_valid': 'bool', 'gps_valid': 'bool', 'target_valid': 'bool', 'gyro_valid': 'bool',
             'timestamp_cyclical_month_sin': 'float32', 'timestamp_cyclical_month_cos': 'float32',
             'timestamp_cyclical_day_sin': 'float32', 'timestamp_cyclical_day_cos': 'float32',
             'timestamp_cyclical_dayofweek_sin': 'float32',
             'timestamp_cyclical_dayofweek_cos': 'float32', 'timestamp_cyclical_hour_sin': 'float32',
             'timestamp_cyclical_hour_cos': 'float32'})

    if get_weather:
        if verbose:
            print("Fetching Weather")
        df_merged = get_weather_data(df_merged)

    if invalid_tags_to_nan:
        if verbose:
            print('Setting invalid tags to nan')
        # set the values for invalid sensors to nan
        df_merged = set_invalids_to_nan(df_merged)

    # Get Surrounging Changes for n_steps
    if get_surrounding_changes:
        if verbose:
            print(f"Retrieving {number_of_timesteps} preceding and following timesteps")
        df_merged = apply_generate_prev_next_features(df_merged,
                                                      n_steps=number_of_timesteps,
                                                      features_list=['gps_lat_m_change',
                                                                     'gps_long_m_change',
                                                                     'gps_speed',
                                                                     'gps_acceleration',
                                                                     'gps_direction',
                                                                     'gps_valid',
                                                                     'gyro_roll',
                                                                     'gyro_pitch',
                                                                     'gyro_yaw',
                                                                     'gyro_valid',
                                                                     'gps_long_speed_from_coords',
                                                                     'gps_lat_speed_from_coords',
                                                                     'gps_abs_speed_from_coords',
                                                                     'gps_lat_accel_from_coords',
                                                                     'gps_long_accel_from_coords',
                                                                     'gps_accel_from_coords'
                                                                     ])

    # calculate standard deviation
    if verbose:
        print('Calculate standard deviation')
    # Calculate the standard deviations for each feature
    gps_lat_m_std = df_merged[df_merged.columns[df_merged.columns.str.contains('gps_lat_m_change')]].std(axis=1).copy()
    gps_long_m_std = df_merged[df_merged.columns[df_merged.columns.str.contains('gps_long_m_change')]].std(axis=1).copy()
    gps_speed_std = df_merged[df_merged.columns[df_merged.columns.str.contains('gps_speed')]].std(axis=1).copy()
    gps_acceleration_std = df_merged[df_merged.columns[df_merged.columns.str.contains('gps_acceleration')]].std(axis=1).copy()
    gyro_roll_std = df_merged[df_merged.columns[df_merged.columns.str.contains('gyro_roll')]].std(axis=1).copy()
    gyro_pitch_std = df_merged[df_merged.columns[df_merged.columns.str.contains('gyro_pitch')]].std(axis=1).copy()
    gyro_yaw_std = df_merged[df_merged.columns[df_merged.columns.str.contains('gyro_yaw')]].std(axis=1).copy()

    # Concatenate the calculated standard deviation values to the original DataFrame
    df_merged = pd.concat([df_merged, gps_lat_m_std.rename('gps_lat_m_std')], axis=1)
    df_merged = pd.concat([df_merged, gps_long_m_std.rename('gps_long_m_std')], axis=1)
    df_merged = pd.concat([df_merged, gps_speed_std.rename('gps_speed_std')], axis=1)
    df_merged = pd.concat([df_merged, gps_acceleration_std.rename('gps_acceleration_std')], axis=1)
    df_merged = pd.concat([df_merged, gyro_roll_std.rename('gyro_roll_std')], axis=1)
    df_merged = pd.concat([df_merged, gyro_pitch_std.rename('gyro_pitch_std')], axis=1)
    df_merged = pd.concat([df_merged, gyro_yaw_std.rename('gyro_yaw_std')], axis=1)

    if drop_nans:
        if verbose:
            print('Dropping NaN-valued Rows')
        df_merged = df_merged.dropna()

    # Train-Test-Split
    if verbose:
        print("Creating Train-Test-Split")
    # shuffle_split
    if shuffle_split:
        unique_ids = df_merged['driveID'].unique()
        np.random.RandomState(42).shuffle(unique_ids)
        idx_split_train = int(len(unique_ids)*train_split)
        train_ids = unique_ids[:idx_split_train]
        test_ids = unique_ids[idx_split_train:]
        df_train = df_merged[df_merged['driveID'].isin(train_ids)]
        df_test = df_merged[df_merged['driveID'].isin(test_ids)]
    else:
        train_split_driveid = train_split * max(df_merged['driveID'])
        df_train = df_merged[df_merged['driveID'] <= train_split_driveid]
        df_test = df_merged[df_merged['driveID'] > train_split_driveid]

    # Save
    if save:
        if verbose:
            print("Pickling Data")

        path_train = os.path.join(savedir, f'{savename}_train.pkl')
        path_test = os.path.join(savedir, f'{savename}_test.pkl')
        df_train.to_pickle(path_train)
        df_test.to_pickle(path_test)

    return df_train, df_test


def dataloader_1hz_trips(filepath="/home/stud03/data_science_challenge/data/whole_trips/huk_telematik_trip_data.sqlite",
                   get_gyro=True, get_accel=True, get_timestamps=False, get_cyclical_time=False,
                   get_weather=False, get_surrounding_changes=True, number_of_timesteps=5, train_split=1.0,
                   shuffle_split=True, invalid_tags_to_nan = False, drop_nans=False,
                   save=False, savedir=None, savename=None, verbose=False, get_coord_accel=True):
    """
     Load (Phone GPS and Telematics Sensor, other phone sensors also included if flags are set accordingly) data from a SQL database containing trips of arbitrary driving situations (no restriction on group_ids),
     transform it to 1Hz and perform further preprocessing steps as indicated by the parameters.

     Parameters:
     - filepath (string): The filepath of the SQL database.
     - get_gyro (bool): Wether data from the phone gyro sensor should be included.
     - get_accel (bool): Wether data from the phone acceleration sensor should be included.
     - get_timestamps (bool): Wether to calculate diverse timestamp-information features from the given timestamps.
     - get_cyclical_time (bool): Wether to also compute cyclical timestamp-features from the features given above.
     - get_weather (bool): Wether to get weather data at the given GPS coordinates and timestamps.
     - get_surrounding_changes (bool): Wether to include the features from the <T> surrounding 1Hz-timestamps into each timestamp.
     - number_of_timestamps (int): <T> feeding into the above parameter.
     - train_split (float): Ratio of train-test-split.
     - shuffle_split (bool): Wether to randomly shuffle the driueIDs before assigning them to either train or test.
     - invalid_tags_to_nan (bool): Wether to transform measurements which are tagged as invalid to NaN.
     - drop_nans (bool): Wether to drop rows with nan values.
     - save (bool): Wether to save the resulting dataframe.
     - savedir (string): The directory to save it in.
     - savename (string): The name to save it as.
     - verbose (bool): Wether to print information about the preprocessing progress in the command line.
     - get_coord_accel (bool): Wether to also calculate the gps-based acceleration as the second derivative of coordinates and not only as the first derivative of speed.

     Returns:
     - df_train (dataframe): DataFrame consisting of 1Hz measurements and calculated features used for training.
     - df_test (dataframe): DataFrame consisting of 1Hz measurements and calculated features used for testing.

     Notes:
     - train_split = 1.0 results in an empty 'test'-df thus enabling crossvalidation on the train-df.
     """
    tqdm.pandas()
    # Startup

    con = sqlite3.connect(filepath)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    cur.fetchall()

    # Phone GPS data
    if verbose:
        print("Reading GPS Data")
    df_features = pd.read_sql_query(f"SELECT driveID, timestamp_utc, osmID, latitude, longitude, direction, speed, gps_status FROM position", con)
    df_features = df_features.rename(
        columns={'latitude': 'gps_lat', 'longitude': 'gps_long', 'direction': 'gps_direction', 'speed': 'gps_speed',
                 'gps_status': 'gps_valid'})
    df_features['gps_valid'] = df_features['gps_valid'].map({'NOT_VALID': 0, 'VALID': 1, 'DATA_MISSING': 0})
    df_features['gps_direction'] = df_features['gps_direction'].map({'FORWARD': 1, 'BACKWARD': -1, 'DATA_MISSING': 0, 'NaN': 0}).fillna(0)
    df_features = df_features.astype(
        {'osmID': 'int32', 'gps_lat': 'float32', 'gps_long': 'float32', 'gps_speed': 'float32',
         'gps_valid': 'int32', 'gps_direction': 'int32'})
    pd.set_option('mode.chained_assignment', None)
    df_features["timestamp_utc"] = [x[:-4] for x in df_features["timestamp_utc"]]
    df_features["timestamp_utc"] = pd.to_datetime(df_features["timestamp_utc"])
    # Calculate Acceleration Based on GPS Data

    if verbose:
        print("Calculating Acceleration from GPS Data")
    df_features = df_features.groupby('driveID').progress_apply(lambda x: calculate_acceleration(x))
    df_features = df_features.reset_index(drop=True)
    # Calculate GPS Mercator Difference between prev and current timestep
    if verbose:
        print("Calculating GPS Change")
    df_features["gps_lat_m"], df_features["gps_long_m"] = convert_coordinates_to_mercator(df_features["gps_lat"], df_features["gps_long"])
    df_features = apply_coordinates_calc_diff(df_features)
    df_features = df_features.reset_index(drop=True)
    if get_coord_accel:
        if verbose:
            print('Calculating speed and accel from gps coords')
        df_features = calculate_accel_from_gps(df_features)
        df_features = df_features.reset_index(drop=True)

    # Phone Gyro Sensor Data
    if get_gyro:
        if verbose:
            print("Reading Gyro Data")
        df_gyro = pd.read_sql_query(f"SELECT driveID, timestamp_utc, roll, pitch, yaw, phone_gyro_status FROM phone_gyro", con)
        df_gyro = df_gyro.rename(
            columns={'roll': 'gyro_roll', 'pitch': 'gyro_pitch', 'yaw': 'gyro_yaw', 'phone_gyro_status': 'gyro_valid'})
        df_gyro['gyro_valid'] = df_gyro['gyro_valid'].map({'NOT_VALID': 0, 'VALID': 1, 'DATA_MISSING': 0})
        df_gyro = df_gyro.astype(
            {'gyro_roll': 'float32', 'gyro_yaw': 'float32', 'gyro_pitch': 'float32',
             'gyro_valid': 'int32'})
        df_gyro['timestamp_utc'] = df_gyro['timestamp_utc'].apply(lambda x: x[:-4])
        # Sample to 1Hz
        df_gyro = df_gyro.groupby("timestamp_utc", as_index=False).agg({
            'driveID': 'first',
            'gyro_roll': 'mean',
            'gyro_pitch': 'mean',
            'gyro_yaw': 'mean',
            'gyro_valid': all_valid
        })
        df_gyro = df_gyro.rename(columns={'driveID first': 'driveID',
                                          'gyro_valid first': 'gyro_valid',
                                          'gyro_valid all_values': 'gyro_valid'})
        df_gyro.columns = df_gyro.columns.str.replace(' ', '_')
        df_gyro["timestamp_utc"] = pd.to_datetime(df_gyro["timestamp_utc"])
        if verbose:
            print('done')
        # Merge to features
        df_features = pd.merge(df_features, df_gyro, on=['driveID', 'timestamp_utc'])
        del df_gyro

    # Phone Acceleration Sensor Data
    if get_accel:
        if verbose:
            print("Reading Accel Data")
        df_accel = pd.read_sql_query(f"SELECT driveID, timestamp_utc, forward, left, up, phone_accel_status FROM phone_accel", con)
        df_accel = df_accel.rename(columns={'forward': 'accel_forward', 'left': 'accel_left', 'up': 'accel_up',
                                            'phone_accel_status': 'accel_valid'})
        df_accel['accel_valid'] = df_accel['accel_valid'].map({'NOT_VALID': 0, 'VALID': 1, 'DATA_MISSING': 0})
        df_accel = df_accel.astype(
            {'accel_forward': 'float32', 'accel_left': 'float32', 'accel_up': 'float32',
             'accel_valid': 'int32'})
        df_accel["timestamp_utc"] = [x[:-4] for x in df_accel["timestamp_utc"]]
        # Sample to 1 Hz
        df_accel = df_accel.groupby("timestamp_utc", as_index=False).agg({
            'driveID': 'first',
            'accel_forward': 'mean',
            'accel_left': 'mean',
            'accel_up': 'mean',
            'accel_valid': all_valid
        })
        df_accel = df_accel.rename(columns={'driveID first': 'driveID',
                                            'accel_valid first': 'accel_valid'})
        df_accel.columns = df_accel.columns.str.replace(' ', '_')
        df_accel["timestamp_utc"] = pd.to_datetime(df_accel["timestamp_utc"])
        # Merge to features
        df_features = pd.merge(df_features, df_accel, on=['driveID', 'timestamp_utc'])

        del df_accel

    # Read target: Car acceleration sensor
    if verbose:
        print("Reading Target Data")
    df_target = pd.read_sql_query(f"SELECT driveId, timestamp_utc, forward, left, up, tag_accel_status FROM tag_accel", con)
    df_target = df_target.rename(columns={'forward': 'target_forward', 'left': 'target_left', 'up': 'target_up',
                                          'tag_accel_status': 'target_valid'})
    df_target['target_valid'] = df_target['target_valid'].map({'NOT_VALID': 0, 'VALID': 1, 'DATA_MISSING': 0})
    df_target = df_target.astype(
        {'target_forward': 'float32', 'target_left': 'float32', 'target_up': 'float32',
         'target_valid': 'int32'})
    df_target["timestamp_utc"] = [x[:-4] for x in df_target["timestamp_utc"]]
    # Sample to 1 Hz
    df_target = df_target.groupby("timestamp_utc", as_index=False).agg({
        'driveID': 'first',
        'target_forward': 'mean',
        'target_left': 'mean',
        'target_up': 'mean',
        'target_valid': all_valid
    })
    df_target = df_target.rename(
        columns={'driveID first': 'driveID', 'target_valid all_valid': 'target_valid'})
    df_target.columns = df_target.columns.str.replace(' ', '_')
    df_target["timestamp_utc"] = pd.to_datetime(df_target["timestamp_utc"])

    # Close SQLite Connection
    con.close()


    # Merge
    if verbose:
        print("Merging Data")
    df_merged = pd.merge(df_features, df_target, on=['driveID', 'timestamp_utc'])

    # Convert valid tags to bools
    column_types = {
        'accel_valid': bool,
        'gps_valid': bool,
        'target_valid': bool,
        'gyro_valid': bool, }

    df_merged = df_merged.astype(column_types)

    # Index driveID
    if verbose:
        print("Indexing DriveID")
    distinct_ids = df_merged['driveID'].unique()
    id_mapping = {id_val: idx for idx, id_val in enumerate(distinct_ids)}
    df_merged['driveID'] = df_merged['driveID'].map(id_mapping)
    df_merged = df_merged.astype({'driveID': 'int32'})

    # Transform timestamps
    if get_timestamps:
        if verbose:
            print("Transforming Timestamps")
        df_merged = add_datepart(df_merged, 'timestamp_utc', time=True, drop=False)
        df_merged.drop(['timestamp_utcIs_month_end', 'timestamp_utcIs_month_start', 'timestamp_utcYear',
                        'timestamp_utcIs_quarter_end', 'timestamp_utcIs_quarter_start', 'timestamp_utcIs_year_end',
                        'timestamp_utcIs_year_start', 'timestamp_utcElapsed', 'timestamp_utcSecond',
                        'timestamp_utcMinute'], axis=1, inplace=True)
        df_merged = df_merged.rename(
            columns={'timestamp_utcMonth': 'timestamp_month', 'timestamp_utcWeek': 'timestamp_week',
                     'timestamp_utcDayofweek': 'timestamp_dayofweek', 'timestamp_utcHour': 'timestamp_hour'})

    # Sinus-Cosinus Encodings
    if get_cyclical_time and get_timestamps:
        if verbose:
            print("Creating Cyclical Encoding")
        df_merged['timestamp_cyclical_month_sin'] = np.sin(2 * np.pi * df_merged['timestamp_month'] / 12.0)
        df_merged['timestamp_cyclical_month_cos'] = np.cos(2 * np.pi * df_merged['timestamp_month'] / 12.0)
        df_merged['timestamp_cyclical_day_sin'] = np.sin(2 * np.pi * df_merged['timestamp_week'] / 7.0)
        df_merged['timestamp_cyclical_day_cos'] = np.cos(2 * np.pi * df_merged['timestamp_week'] / 7.0)
        df_merged['timestamp_cyclical_dayofweek_sin'] = np.sin(2 * np.pi * df_merged['timestamp_dayofweek'] / 7.0)
        df_merged['timestamp_cyclical_dayofweek_cos'] = np.cos(2 * np.pi * df_merged['timestamp_dayofweek'] / 7.0)
        df_merged['timestamp_cyclical_hour_sin'] = np.sin(2 * np.pi * df_merged['timestamp_hour'] / 24.0)
        df_merged['timestamp_cyclical_hour_cos'] = np.cos(2 * np.pi * df_merged['timestamp_hour'] / 24.0)
        df_merged = df_merged.astype(
            {'accel_valid': 'bool', 'gps_valid': 'bool', 'target_valid': 'bool', 'gyro_valid': 'bool',
             'timestamp_cyclical_month_sin': 'float32', 'timestamp_cyclical_month_cos': 'float32',
             'timestamp_cyclical_day_sin': 'float32', 'timestamp_cyclical_day_cos': 'float32',
             'timestamp_cyclical_dayofweek_sin': 'float32',
             'timestamp_cyclical_dayofweek_cos': 'float32', 'timestamp_cyclical_hour_sin': 'float32',
             'timestamp_cyclical_hour_cos': 'float32'})

    if get_weather:
        if verbose:
            print("Fetching Weather")
        df_merged = get_weather_data(df_merged)

    if invalid_tags_to_nan:
        if verbose:
            print('Setting invalid tags to nan')
        # set the values for invalid sensors to nan
        df_merged = set_invalids_to_nan(df_merged)

    # Get Surrounging Changes for n_steps
    if get_surrounding_changes:
        if verbose:
            print(f"Retrieving {number_of_timesteps} preceding and following timesteps")
        df_merged = apply_generate_prev_next_features(df_merged,
                                                      n_steps=number_of_timesteps,
                                                      features_list=['gps_lat_m_change',
                                                                     'gps_long_m_change',
                                                                     'gps_speed',
                                                                     'gps_acceleration',
                                                                     'gps_direction',
                                                                     'gps_valid',
                                                                     'gyro_roll',
                                                                     'gyro_pitch',
                                                                     'gyro_yaw',
                                                                     'gyro_valid',
                                                                     'gps_long_speed_from_coords',
                                                                     'gps_lat_speed_from_coords',
                                                                     'gps_abs_speed_from_coords',
                                                                     'gps_lat_accel_from_coords',
                                                                     'gps_long_accel_from_coords',
                                                                     'gps_accel_from_coords'
                                                                     ])

    # calculate standard deviation
    if verbose:
        print('Calculate standard deviation')
    # Calculate the standard deviations for each feature
    gps_lat_m_std = df_merged[df_merged.columns[df_merged.columns.str.contains('gps_lat_m_change')]].std(axis=1).copy()
    gps_long_m_std = df_merged[df_merged.columns[df_merged.columns.str.contains('gps_long_m_change')]].std(axis=1).copy()
    gps_speed_std = df_merged[df_merged.columns[df_merged.columns.str.contains('gps_speed')]].std(axis=1).copy()
    gps_acceleration_std = df_merged[df_merged.columns[df_merged.columns.str.contains('gps_acceleration')]].std(axis=1).copy()
    gyro_roll_std = df_merged[df_merged.columns[df_merged.columns.str.contains('gyro_roll')]].std(axis=1).copy()
    gyro_pitch_std = df_merged[df_merged.columns[df_merged.columns.str.contains('gyro_pitch')]].std(axis=1).copy()
    gyro_yaw_std = df_merged[df_merged.columns[df_merged.columns.str.contains('gyro_yaw')]].std(axis=1).copy()

    # Concatenate the calculated standard deviation values to the original DataFrame
    df_merged = pd.concat([df_merged, gps_lat_m_std.rename('gps_lat_m_std')], axis=1)
    df_merged = pd.concat([df_merged, gps_long_m_std.rename('gps_long_m_std')], axis=1)
    df_merged = pd.concat([df_merged, gps_speed_std.rename('gps_speed_std')], axis=1)
    df_merged = pd.concat([df_merged, gps_acceleration_std.rename('gps_acceleration_std')], axis=1)
    df_merged = pd.concat([df_merged, gyro_roll_std.rename('gyro_roll_std')], axis=1)
    df_merged = pd.concat([df_merged, gyro_pitch_std.rename('gyro_pitch_std')], axis=1)
    df_merged = pd.concat([df_merged, gyro_yaw_std.rename('gyro_yaw_std')], axis=1)

    if drop_nans:
        if verbose:
            print('Dropping NaN-valued Rows')
        df_merged = df_merged.dropna()

    # Train-Test-Split
    if verbose:
        print("Creating Train-Test-Split")
    # shuffle_split
    if shuffle_split:
        unique_ids = df_merged['driveID'].unique()
        np.random.RandomState(42).shuffle(unique_ids)
        idx_split_train = int(len(unique_ids)*train_split)
        train_ids = unique_ids[:idx_split_train]
        test_ids = unique_ids[idx_split_train:]
        df_train = df_merged[df_merged['driveID'].isin(train_ids)]
        df_test = df_merged[df_merged['driveID'].isin(test_ids)]
    else:
        train_split_driveid = train_split * max(df_merged['driveID'])
        df_train = df_merged[df_merged['driveID'] <= train_split_driveid]
        df_test = df_merged[df_merged['driveID'] > train_split_driveid]

    # Save
    if save:
        if verbose:
            print("Pickling Data")
        # df_train.to_pickle(f'../../data/{savename}_1hz_train.pkl')
        # df_test.to_pickle(f'../../data/{savename}_1hz_test.pkl')

        path_train = os.path.join(savedir, f'{savename}_train.pkl')
        path_test = os.path.join(savedir, f'{savename}_test.pkl')
        df_train.to_pickle(path_train)
        df_test.to_pickle(path_test)

    return df_train, df_test


def generate_data(path_sql, save_directory):
    """
    Generate and save processed data from the SQLite database for specific group IDs.

    This function loads the data from an SQLite database using the `dataloader_1hz` function, processes it,
    and then saves the resulting data for each group ID in a specified directory.

    Parameters:
    - path_sql (str): Path to the SQLite database file.
    - save_directory (str): Directory path where the processed data will be saved.

    Returns:
    None. Processed data is saved to the specified directory.
    """

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    group_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for group_id in group_ids:
        df_train, _ = dataloader_1hz(filepath=path_sql, group_id=group_id, get_gyro=True, get_accel=True,
                                     get_timestamps=False,
                                     get_cyclical_time=False, get_weather=False, get_surrounding_changes=True,
                                     number_of_timesteps=5, train_split=1, shuffle_split=True, min_entry_driveid=10,
                                     invalid_tags_to_nan=True, drop_nans=False,
                                     max_amount=None, save=True, savedir=save_directory,
                                     savename=f'1hz_group_{group_id}', verbose=False, get_coord_accel=True)


def load_data(save_dir, dropna=True, drop_accel_invalid=True, drop_gyro_invalid=True, drop_gps_invalid=True):
    """
    Load and preprocess data from pickle files saved in a specified directory.

    Parameters:
    - save_dir (str): Directory path where the pickle files are saved.
    - dropna (bool, optional): If True, drops rows with any NaN values. Defaults to True.
    - drop_accel_invalid (bool, optional): If True, removes drive IDs with invalid accelerometer readings. Defaults to True.
    - drop_gyro_invalid (bool, optional): If True, removes drive IDs with invalid gyroscope readings. Defaults to True.
    - drop_gps_invalid (bool, optional): If True, removes drive IDs with invalid GPS readings. Defaults to True.

    Returns:
    - df (DataFrame): The preprocessed dataframe containing the loaded data.

    Notes:
    The function specifically expects files with names '1hz_group_x_train.pkl' where x ranges from 1 to 9.
    """

    # get data
    df1 = pd.read_pickle(os.path.join(save_dir, '1hz_group_1_train.pkl'))
    df2 = pd.read_pickle(os.path.join(save_dir, '1hz_group_2_train.pkl'))
    df3 = pd.read_pickle(os.path.join(save_dir, '1hz_group_3_train.pkl'))
    df4 = pd.read_pickle(os.path.join(save_dir, '1hz_group_4_train.pkl'))
    df5 = pd.read_pickle(os.path.join(save_dir, '1hz_group_5_train.pkl'))
    df6 = pd.read_pickle(os.path.join(save_dir, '1hz_group_6_train.pkl'))
    df7 = pd.read_pickle(os.path.join(save_dir, '1hz_group_7_train.pkl'))
    df8 = pd.read_pickle(os.path.join(save_dir, '1hz_group_8_train.pkl'))
    df9 = pd.read_pickle(os.path.join(save_dir, '1hz_group_9_train.pkl'))

    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9], axis=0)
    df['driveID'] = pd.factorize(df['driveID'].astype(str) + df['groupID'].astype(str))[0]
    df = df.reset_index(drop=True)

    del df1, df2, df3, df4, df5, df6, df7, df8, df9

    # replace values
    df = df.replace({True: 1, False: 0})

    # drop valid_ features
    df = df.drop(columns=df.columns[df.columns.str.contains('valid_')])

    if dropna:
        print(df.shape)
        df.dropna(inplace=True)
        print(df.shape)

    if drop_accel_invalid:
        corrupt_drive_ids = list(set(df[df['accel_valid'] == 0]['driveID']))
        df = df[~df['driveID'].isin(corrupt_drive_ids)]
        print(df.shape)
    if drop_gyro_invalid:
        corrupt_drive_ids = list(set(df[df['gyro_valid'] == 0]['driveID']))
        df = df[~df['driveID'].isin(corrupt_drive_ids)]
        print(df.shape)
    if drop_gps_invalid:
        corrupt_drive_ids = list(set(df[df['gps_valid'] == 0]['driveID']))
        df = df[~df['driveID'].isin(corrupt_drive_ids)]
        print(df.shape)

    # Count the number of entries for each driveID
    drive_counts = df['driveID'].value_counts()

    # Get the driveIDs with at least 15 entries
    valid_drive_ids = drive_counts[drive_counts >= 15].index

    # Filter the DataFrame based on the valid driveIDs
    df = df[df['driveID'].isin(valid_drive_ids)]

    return df

