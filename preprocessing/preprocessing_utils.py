import numpy as np
from meteostat import Point, Hourly
from datetime import timedelta
from functools import partial
from pyproj import Transformer
import multiprocessing
from math import sin, cos
import pandas as pd
from tqdm import tqdm


def all_valid(x):
    """
    Check if all elements in a list or array are valid (equal to 1).

    Parameters:
    - x (list or array-like): A list or array-like object containing numerical elements.

    Returns:
    - int: Returns 1 if all elements are equal to 1, otherwise returns 0.
    """
    if sum(x) == len(x):
        return 1
    else:
        return 0


def hour_rounder(t):
    """
    Rounds a datetime object to the nearest hour.

    Parameters:
    - t (datetime): Input datetime object to be rounded.

    Returns:
    - datetime: Rounded datetime object.
    """
    return t.replace(second=0, microsecond=0, minute=0, hour=t.hour) + timedelta(hours=t.minute // 30)


def get_weather_data(df):
    """
    add weather data based on GPS coordinates and timestamps to the input data.

    Parameters:
    - df (pd.DataFrame): A dataframe containing columns 'timestamp_utc', 'gps_lat', and 'gps_long'.
                         'timestamp_utc' should be a datetime column, while 'gps_lat' and 'gps_long'
                         should contain latitude and longitude values, respectively.

    Returns:
    - pd.DataFrame: The input dataframe with additional columns containing weather data for
                    each GPS coordinate and timestamp.

    Notes:
    - This function computes the center of all GPS coordinates and rounds the 'timestamp_utc' to the nearest hour.
    - It fetches the hourly weather data for the center GPS coordinate and rounded timestamps using the Hourly API.
    - The resultant dataframe contains the original columns of the input dataframe, plus 'weather_precipitation'
      and 'weather_windspeed' columns derived from the fetched weather data.
    - The function utilizes the `tqdm` library to display a progress bar when fetching weather data. Ensure
      the 'tqdm' library is imported before using this function.
    - Before using this function, you need to ensure that necessary API calls or methods, such as `Hourly` and
      `Point`, are imported and properly set up.
    """
    # Get center point of group
    lat_center = df['gps_lat'].mean()
    long_center = df['gps_long'].mean()

    # Get timepoints
    df['time'] = df['timestamp_utc'].apply(hour_rounder)
    # lambda x: x.replace(second=0, microsecond=0, minute=0, hour=x.hour) + timedelta(hours=x.minute // 30))
    min_timestamp = df['time'].min()
    max_timestamp = df['time'].max()
    current_hour = min_timestamp
    all_timestamps = []
    while current_hour < max_timestamp:
        all_timestamps.append(current_hour)
        current_hour += timedelta(hours=1)

    # Get weather data from API
    coordinate = Point(lat_center, long_center)
    weather_df = pd.DataFrame()
    for time in tqdm(all_timestamps):
        weather = Hourly(coordinate, time, time).fetch()
        new_row = pd.Series([time, weather])
        weather_df = pd.concat([weather_df, new_row[1]])

    # Merge
    df = pd.merge(df, weather_df, on='time', how='left')
    df = df.rename(columns={'prcp': 'weather_precipitation', 'wspd': 'weather_windspeed'})
    df = df.drop(['time', 'temp', 'dwpt', 'rhum', 'snow', 'wdir', 'wpgt', 'pres', 'tsun', 'coco'],
                 axis=1)  # Drop irrelevant features

    return df


def calculate_acceleration(df):
    """
    Calculate the acceleration based on GPS speed and timestamps in the dataframe.

    Parameters:
    - df (pd.DataFrame): A dataframe containing columns 'timestamp_utc' and 'gps_speed'.
                         'timestamp_utc' should be a datetime column, while 'gps_speed'
                         should contain speed values (in any consistent unit).

    Returns:
    - pd.DataFrame: A modified dataframe with an additional column 'gps_acceleration'
                    which represents the change in speed over the change in time.

    Notes:
    - The function computes acceleration as the difference in consecutive GPS speeds
      divided by the time interval between those data points.
    - The first row of the resulting dataframe will be dropped since it doesn't have a
      previous data point to compute the acceleration.
    - Although there is commented code for smoothing the acceleration using the
      Savitzky-Golay filter, this functionality is currently not active.
    """

    # Calculate Time Differences
    df['time_interval_in_s'] = (df['timestamp_utc'] - df['timestamp_utc'].shift(1)).dt.total_seconds()

    # Calculate Acceleration
    df['gps_acceleration'] = (df['gps_speed'] - df['gps_speed'].shift(1)) / df['time_interval_in_s']

    # Drop NaN Row And Time Interval Column
    df = df.iloc[1:].copy()
    df = df.drop('time_interval_in_s', axis=1)

    # Smooth Acceleration
    # Window length and Poly Order must meet these requirements
    # filter_window = min(20, len(df['gps_acceleration']))
    # filter_poly = min(4, filter_window-1)
    # df['gps_acceleration_smoothed'] = savgol_filter(df['gps_acceleration'], window_length=filter_window, polyorder=filter_poly)

    return df


def convert_coordinates_to_mercator(lat_series, lon_series):
    """
    Convert geographic coordinates (latitude and longitude) to Web Mercator coordinates.

    Parameters:
    - lat_series (pd.Series): A Pandas Series containing latitude values.
    - lon_series (pd.Series): A Pandas Series containing longitude values.

    Returns:
    - tuple: A tuple containing two lists:
        1. List of x-coordinates (in meters) in Web Mercator projection.
        2. List of y-coordinates (in meters) in Web Mercator projection.

    Notes:
    - This function utilizes the pyproj library to transform geographic coordinates
      from the standard WGS 84 (EPSG:4326) to the Web Mercator (EPSG:3857) projection.
    """

    # Create a transformer object for coordinate transformation
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")

    # Convert the GPS coordinates to Mercator projection
    mercator_coordinates = transformer.transform(lon_series.tolist(), lat_series.tolist())

    # Create a new DataFrame with Mercator coordinates
    lat_series = mercator_coordinates[0]
    lat_long = mercator_coordinates[1]

    return lat_series, lat_long


def generate_prev_next_features(df, n_steps, features_list):
    """
    Generates previous and upcoming features for each specified feature in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        n_steps (int): The number of steps to look back and look ahead.
        features_list (list): A list of column names representing the features.

    Returns:
        pd.DataFrame: The modified DataFrame with additional columns for previous and upcoming features.

    Example:
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        generate_prev_next_features(df, 2, ['A'])

        Output:
            A  A_-1  A_-2  A_+1  A_+2
        0  1   NaN   NaN   2.0   3.0
        1  2   1.0   NaN   3.0   4.0
        2  3   2.0   1.0   4.0   5.0
        3  4   3.0   2.0   5.0   NaN
        4  5   4.0   3.0   NaN   NaN
    """
    df_last = pd.DataFrame()
    df_upcoming = pd.DataFrame()

    # Retrieve the changes of the features for the last n steps (t-1, t-2, ..., t-n and t+1, t+2, ..., t+n)
    for column in features_list:
        last_cols = pd.concat([df[column].shift(i) for i in range(1, n_steps + 1)], axis=1)
        last_cols.columns = [f'{column}_-{i}' for i in range(1, n_steps + 1)]
        df_last = pd.concat([df_last, last_cols], axis=1)

        upcoming_cols = pd.concat([df[column].shift(-i) for i in range(1, n_steps + 1)], axis=1)
        upcoming_cols.columns = [f'{column}_+{i}' for i in range(1, n_steps + 1)]
        df_upcoming = pd.concat([df_upcoming, upcoming_cols], axis=1)

    return pd.concat([df, df_last, df_upcoming], axis=1)


def apply_generate_prev_next_features(df, n_steps, features_list):
    """
    Applies the generation of previous and upcoming features (timesteps) to each driveID in a DataFrame using parallel processing.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        n_steps (int): The number of steps to look back and look ahead.
        features_list (list): A list of column names representing the features.

    Returns:
        pd.DataFrame: The modified DataFrame with additional columns for previous and upcoming features.
    """
    tqdm.pandas()
    # Number of processes to use for parallelization
    num_processes = multiprocessing.cpu_count()
    # Split the DataFrame into groups based on 'driveID'
    grouped = df.groupby('driveID')
    # Create a partial function for generate_prev_next_features
    partial_generate_prev_next_features = partial(generate_prev_next_features, n_steps=n_steps,
                                                  features_list=features_list)
    # Create a multiprocessing pool
    pool = multiprocessing.Pool(num_processes)

    # Apply the function to each group in parallel
    results = list(
        tqdm(pool.imap(partial_generate_prev_next_features, [group for _, group in grouped]), total=len(grouped)))

    # Close the pool to release resources
    pool.close()
    pool.join()

    # Combine the results into a DataFrame
    df_combined = pd.concat(results)

    return df_combined


def set_invalids_to_nan(data):
    """
    Replace sensor readings marked as invalid with NaN values.

    For each sensor type (GPS, Gyro, and Accel), this function identifies the columns associated
    with that sensor type in the dataframe. It then checks the corresponding 'valid' column
    ('gps_valid', 'gyro_valid', 'accel_valid'). For any row where the 'valid' column is set to 0,
    the function replaces the actual sensor reading with a NaN value.

    Parameters:
    - data (pd.DataFrame): The input dataframe containing sensor readings and their associated validity tags.

    Returns:
    - pd.DataFrame: A modified dataframe where invalid sensor readings are replaced with NaN values.

    Notes:
    - The 'valid' columns ('gyro_valid', 'accel_valid', 'gps_valid') are used to indicate the validity of sensor readings.
      A value of 0 in these columns indicates invalid data, while a value of 1 indicates valid data.
    - Columns with sensor data are identified based on their prefixes ('gps_', 'gyro_', 'accel_'). The 'valid' columns
      are excluded from this identification.
    """
    gps_columns = list(data.columns[data.columns.str.contains('gps_') & (data.columns != 'gps_valid')])
    gyro_columns = list(data.columns[data.columns.str.contains('gyro_') & (data.columns != 'gyro_valid')])
    accel_columns = list(data.columns[data.columns.str.contains('accel_') & (data.columns != 'accel_valid')])

    data.loc[data['gps_valid'] == 0, gps_columns] = np.nan
    data.loc[data['gyro_valid'] == 0, gyro_columns] = np.nan
    data.loc[data['accel_valid'] == 0, accel_columns] = np.nan

    return data


def get_radius_at_lat(lat):
    """
    Compute the Earth's radius at a given latitude based on the WGS-84 ellipsoid model.

    Parameters:
    - lat (float or np.array): Latitude in degrees. Can be a scalar value or an array of latitudes.

    Returns:
    - float or np.array: Earth's radius in meters at the given latitude(s).

    Notes:
    - The Earth is not a perfect sphere but an oblate spheroid, meaning it's slightly flattened at the poles
      and bulging at the equator.
    - The WGS-84 (World Geodetic System 1984) is a widely used standard for the Earth's ellipsoid model.
      The `a` and `b` values in the function are the semi-major and semi-minor axis lengths respectively,
      as defined by WGS-84.
    - The returned radius will be maximum at the equator and will decrease as the latitude approaches the poles.
    """

    lat_radian = np.radians(lat)

    a = 6378137.0  # Calculations in meters!
    b = 6356752.314

    f1 = pow((pow(a, 2) * cos(lat_radian)), 2)
    f2 = pow((pow(b, 2) * sin(lat_radian)), 2)
    f3 = pow((a * cos(lat_radian)), 2)
    f4 = pow((b * sin(lat_radian)), 2)

    radius = np.sqrt((f1 + f2) / (f3 + f4))

    return radius


def calculate_coordinate_diff(df):
    """
    Calculate the change in coordinates (latitude and longitude) to the preceding timestamp using Mercator projections.
    """

    # Calculate change in coordinates to the preceding timestamp
    df['gps_lat_m_change'] = df['gps_lat_m'] - df['gps_lat_m'].shift(1)
    df['gps_long_m_change'] = df['gps_long_m'] - df['gps_long_m'].shift(1)

    return df


def calculate_coordinate_diff_no_mercator(df):
    """
    Calculate the change in coordinates (latitude and longitude) to the preceding timestamp.
    """

    # Calculate change in coordinates to the preceding timestamp
    df['gps_lat_change'] = df['gps_lat'] - df['gps_lat'].shift(1)
    df['gps_long_change'] = df['gps_long'] - df['gps_long'].shift(1)

    return df


def apply_coordinates_calc_diff(df):
    """
    Applies the calculation of coordinate differences to each driveID in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The modified DataFrame with additional columns for coordinate differences.
    """

    tqdm.pandas()

    # Apply the calculate_gps_difference function to each group in the DataFrame
    df = df.groupby('driveID').progress_apply(lambda x: calculate_coordinate_diff(x))

    return df


def calculate_accel_from_gps(df):
    """
    Calculates acceleration based on GPS coordinates of a vehicle's journey.

    Given a DataFrame with GPS latitude and longitude data per timestamp, this function computes the vehicle's
    latitudinal, longitudinal, and absolute acceleration using differences in position and speed. It uses the mean
    latitude to determine the Earth's radius at that location, which is essential for accurately calculating
    distances based on latitudinal and longitudinal changes.

    Parameters:
    - df_train (pd.DataFrame): Input DataFrame containing at least 'gps_lat', 'gps_long', and 'driveID' columns.
      - gps_lat (float): Latitude coordinates.
      - gps_long (float): Longitude coordinates.
      - driveID (int or str): Unique identifier for each drive session.

    Returns:
    - pd.DataFrame: DataFrame with added columns for latitudinal, longitudinal, and absolute speed and acceleration:
      - gps_long_speed_from_coords (float): Speed based on longitudinal coordinate changes (m/s).
      - gps_lat_speed_from_coords (float): Speed based on latitudinal coordinate changes (m/s).
      - gps_abs_speed_from_coords (float): Absolute speed computed using Pythagoras' theorem (m/s).
      - gps_lat_accel_from_coords (float): Acceleration in the latitudinal direction (m/s^2).
      - gps_long_accel_from_coords (float): Acceleration in the longitudinal direction (m/s^2).
      - gps_accel_from_coords (float): Absolute acceleration (m/s^2).

    Notes:
    - Assumes the motion of the car is mostly on the plane tangential to the Earth's surface.
    - The calculated values are approximations due to the inherent inaccuracies in GPS measurements and not
      accounting for altitude changes or the vehicle's orientation.
    """

    # Get mean latitude for calculation of radius
    lat_mean = df['gps_lat'].mean()

    # Calculate radius of earth at given lat
    r = get_radius_at_lat(lat_mean)

    # Calculate coord diffs
    df = df.groupby('driveID').apply(lambda x: calculate_coordinate_diff_no_mercator(x))  # given in degrees

    # Calculate velocity in lat and long direction
    df['gps_long_speed_from_coords'] = np.radians(df['gps_long_change']) * r  # given in meters/s!
    df['gps_lat_speed_from_coords'] = np.radians(df['gps_lat_change']) * r  # given in meters/s!

    # Calculate absolute velocity
    df['gps_abs_speed_from_coords'] = np.sqrt(df['gps_long_speed_from_coords'] ** 2 + df[
        'gps_lat_speed_from_coords'] ** 2)  # Achtung! Das ist eine absolute Größe, unabhängig der Fahrtrichtung, da diese Info nicht in den GPS steckt (Auto könnte auch rückwärts fahren)

    # Calculate acceleration in lat and long direction
    df['gps_lat_accel_from_coords'] = df['gps_lat_speed_from_coords'] - df[
        'gps_lat_speed_from_coords'].shift(1)
    df['gps_long_accel_from_coords'] = df['gps_long_speed_from_coords'] - df[
        'gps_long_speed_from_coords'].shift(1)

    # Calculate absolute accel
    df['gps_accel_from_coords'] = df['gps_abs_speed_from_coords'] - df[
        'gps_abs_speed_from_coords'].shift(
        1)  # Diese Größe beschreibt Bremsen/Beschleunigen der absoluten Geschwindigkeit, könnte also auch umgekehrt sein, diese Info haben wir hier nicht

    df.drop(axis=1, columns=['gps_long_change', 'gps_lat_change'], inplace=True)

    return df
