import os
import sys
import streamlit as st
import pandas as pd
from dataclasses import dataclass

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PROJECT_PATH)

from model.meta_learner import get_results, plot_driveIDs2


@dataclass
class DataOverview:
    """
    A class to encapsulate data details for overview display.
    """
    df_y: any
    target_name: str


@dataclass
class DataAnalysis:
    """
    A class to encapsulate data details for the analysis.
    """
    trip_lengths: any
    trip_max_accels: any
    sensor_errors: any
    sensor_errors_f: any


@dataclass
class SidebarConfig:
    """
    Configuration for the sidebar in the Streamlit app.

    This class encapsulates the various configuration options and data
    settings required for rendering and controlling sidebar elements in
    the Streamlit application.
    """

    df_y_filtered: any
    target_name: str
    metric_model: any
    metric: any
    typ: any
    random_state: int
    model_list: any
    input_driveID: any
    rows: int
    cols: int
    figsize_width: int
    figsize_height: int


def configure_layout():
    """Set up the Streamlit layout configuration."""
    st.set_page_config(layout="wide")
    st.title('Vehicle Acceleration Modelling - Visualization')
    return st.file_uploader("Please upload the model predictions", type="pkl")


@st.cache_data
def load_and_process_data(uploaded_file):
    """
    Load, process, and extract relevant columns and metrics from the provided pickle file.

    Parameters:
    - uploaded_file (IO): The input file object containing the pickled dataframe.

    Returns:
    - df_y (pd.DataFrame): The loaded dataframe from the pickle file.
    - model_list_input (list[str]): A list of column names containing predictions.
    - targets_available (list[str]): A list of available target columns in the dataframe.
    - df_results (pd.DataFrame): A dataframe containing results for specific targets.
    """

    # Load data from the pickled file
    df_y = pd.read_pickle(uploaded_file)

    # Extract prediction and target columns
    model_list_input = [colname for colname in df_y.columns if 'pred' in colname]
    targets_available = [colname for colname in df_y.columns if 'target' in colname]

    # Remove 'target_valid' from the list of available targets
    targets_available.remove('target_valid')

    # This constant is needed to calculate the out-of-sample r^2. For convenience purposes it is set to 0
    y_train_avg = 0

    # Calculate metrics based on the availability of certain target columns
    if 'target_forward' in targets_available:
        df_results = get_results(df_y, df_y[['gps_speed']], 'target_forward', y_train_avg, meta_only=False)
    elif 'target_left' in targets_available:
        df_results = get_results(df_y, df_y[['gps_speed']], 'target_left', y_train_avg, meta_only=False)

    return df_y, model_list_input, targets_available, df_results


def select_metrics(typ, model_list_input):
    """
    Determine the metrics and related parameters based on the provided type.

    Based on the provided 'typ', this function uses Streamlit widgets to get user
    input for the model and metric of interest. It also determines the random state
    based on the type.

    Parameters:
    - typ (str): The type of metrics selection. Valid options are 'best', 'worst', or 'random'.
    - model_list_input (list[str]): A list of models from which the user can select.

    Returns:
    - metric_model (str): The chosen model's name.
    - metric (str): The chosen metric's name. Valid metrics are 'mae', 'mse', 'mape', or 'r2'.
    - random_state (int): The determined random seed.

    Notes:
    - The function assumes that Streamlit (st) is available in the namespace.
    - Default values for 'metric_model' and 'metric' are 'meta_model_pred' and 'mae' respectively
      if the type is neither 'best' nor 'worst'.
    - Default value for 'random_state' is 42 unless the type is 'random'.
    """

    # Determine model and metric based on type
    if typ in ['best', 'worst']:
        metric_model = st.selectbox('Model (considered at Type)', model_list_input)
        metric = st.selectbox('Metric', ['mae', 'mse', 'mape', 'r2'])
    else:
        metric_model, metric = 'meta_model_pred', 'mae'

    # Determine random state based on type
    if typ == 'random':
        random_state = st.slider('Random Seed', min_value=1, max_value=200, value=100)
    else:
        random_state = 42

    return metric_model, metric, random_state


def analyze_data(df_y, target_name):
    """Analyze the dataset to derive various attributes."""
    trip_lengths = df_y[['driveID', 'timestamp_utc']].groupby('driveID').agg('count')

    df_accels = df_y[['driveID', target_name]].copy()
    df_accels[target_name] = abs(df_accels[target_name])
    trip_max_accels = df_accels.groupby('driveID').agg('max')

    sensor_errors = df_y[['driveID', 'gps_valid', 'gyro_valid', 'accel_valid']].groupby('driveID').apply(
        lambda x: (x == 0).sum())
    sensor_errors_f = df_y[['driveID', 'gps_valid', 'gyro_valid', 'accel_valid']].groupby('driveID').apply(
        lambda x: (x == 0).sum() / len(x))

    return trip_lengths, trip_max_accels, sensor_errors, sensor_errors_f


def sidebar_configuration(data_overview, targets_available, model_list_input, data_analysis):
    """Configure the Streamlit sidebar for user input and return the filtered dataframe."""
    with st.sidebar:
        # Plot Parameters
        st.title(f'Plot Parameters')
        st.header('Target and Models')

        model_and_target = targets_available + model_list_input
        target_name = data_overview.target_name

        model_list = st.multiselect('What To Plot', options=model_and_target, default=[target_name, 'meta_model_pred'])

        # Trip Selection
        st.header('Trip Selection')

        shortest_trip, longest_trip = min(data_analysis.trip_lengths['timestamp_utc']), max(data_analysis.trip_lengths['timestamp_utc'])
        lowest_accel, highest_accel = min(data_analysis.trip_max_accels[target_name]), max(data_analysis.trip_max_accels[target_name])
        min_length, max_length = st.slider('Length (s)', min_value=shortest_trip, max_value=longest_trip,
                                           value=(shortest_trip, longest_trip))
        min_accel, max_accel = st.slider('Max absolute Acceleration (|m/s^2|)', min_value=lowest_accel,
                                         max_value=highest_accel, value=(lowest_accel, highest_accel))

        typ = st.selectbox('Type', ['random', 'best', 'worst'])
        metric_model, metric, random_state = select_metrics(typ, model_list_input)

        # Invalid Sensors
        st.header('Invalid Sensors')


        if max(data_analysis.sensor_errors['gps_valid']) > min(data_analysis.sensor_errors['gps_valid']):
            min_inval_gps, max_inval_gps = st.slider('GPS: Number of invalid sensor values',
                                                     min_value=min(data_analysis.sensor_errors['gps_valid']),
                                                     max_value=max(data_analysis.sensor_errors['gps_valid']),
                                                     value=(min(data_analysis.sensor_errors['gps_valid']), max(data_analysis.sensor_errors['gps_valid'])))

            min_inval_gps_f, max_inval_gps_f = st.slider('GPS: Fraction of invalid sensor values',
                                                         min_value=min(data_analysis.sensor_errors_f['gps_valid']),
                                                         max_value=max(data_analysis.sensor_errors_f['gps_valid']),
                                                         value=(min(data_analysis.sensor_errors_f['gps_valid']), max(data_analysis.sensor_errors_f['gps_valid'])))

            drive_ids_w_sensor_err_gps = data_analysis.sensor_errors[
                (data_analysis.sensor_errors['gps_valid'] >= min_inval_gps) & (data_analysis.sensor_errors['gps_valid'] <= max_inval_gps)].index
            drive_ids_w_sensor_err_gps_f = data_analysis.sensor_errors_f[(data_analysis.sensor_errors_f['gps_valid'] >= min_inval_gps_f) & (
                    data_analysis.sensor_errors_f['gps_valid'] <= max_inval_gps_f)].index

        else:

            drive_ids_w_sensor_err_gps = data_overview.df_y['driveID']
            drive_ids_w_sensor_err_gps_f = data_overview.df_y['driveID']

        if max(data_analysis.sensor_errors['gyro_valid']) > min(data_analysis.sensor_errors['gyro_valid']):
            min_inval_gyro, max_inval_gyro = st.slider('Gyro and Accel: Number of invalid sensor values',
                                                       min_value=min(data_analysis.sensor_errors['gyro_valid']),
                                                       max_value=max(data_analysis.sensor_errors['gyro_valid']),
                                                       value=(min(data_analysis.sensor_errors['gyro_valid']), max(data_analysis.sensor_errors['gyro_valid'])))

            min_inval_gyro_f, max_inval_gyro_f = st.slider('Gyro and Accel: Fraction of invalid sensor values',
                                                           min_value=min(data_analysis.sensor_errors_f['gyro_valid']),
                                                           max_value=max(data_analysis.sensor_errors_f['gyro_valid']),
                                                           value=(min(data_analysis.sensor_errors_f['gyro_valid']),
                                                                  max(data_analysis.sensor_errors_f['gyro_valid'])))

            drive_ids_w_sensor_err_gyro = data_analysis.sensor_errors[
                (data_analysis.sensor_errors['gyro_valid'] >= min_inval_gyro) & (data_analysis.sensor_errors['gyro_valid'] <= max_inval_gyro)].index
            drive_ids_w_sensor_err_gyro_f = data_analysis.sensor_errors_f[(data_analysis.sensor_errors_f['gyro_valid'] >= min_inval_gyro_f) & (
                    data_analysis.sensor_errors_f['gyro_valid'] <= max_inval_gyro_f)].index

        else:
            drive_ids_w_sensor_err_gyro = data_overview.df_y['driveID']
            drive_ids_w_sensor_err_gyro_f = data_overview.df_y['driveID']

        #print(drive_ids_w_sensor_err_gps)

        drive_ids_w_sensor_err = set(drive_ids_w_sensor_err_gps) & set(drive_ids_w_sensor_err_gyro) & set(
            drive_ids_w_sensor_err_gps_f) & set(drive_ids_w_sensor_err_gyro_f)



        # Data Filtering based on Sidebar Selections
        df_y_filtered = filter_dataframe_based_on_sidebar(data_overview.df_y, data_analysis.trip_lengths, data_analysis.trip_max_accels, target_name, min_length, max_length, min_accel, max_accel, drive_ids_w_sensor_err)

        # The title of the section
        st.header('Specific Trip')

        # The checkbox acts as a toggle to show/hide content
        if st.checkbox('Show Specific Trips Options', value=False):
            driveIDs = df_y_filtered['driveID'].copy()
            driveIDs.reset_index(drop=True, inplace=True)
            index = st.slider('Select a DriveID index', 0, len(driveIDs) - 1)
            selected_driveID = driveIDs[index]
            input_driveID = [st.text_input('DriveID', value=selected_driveID)]
        else:
            input_driveID = None

        # Customization
        st.header('Customization')
        rows, cols = st.slider('Rows', min_value=1, max_value=10, value=1), st.slider('Cols', min_value=1, max_value=10,
                                                                                      value=1)
        figsize_width, figsize_height = st.slider('Width', min_value=1, max_value=40, value=10), st.slider('Height',
                                                                                                           min_value=1,
                                                                                                           max_value=40,
                                                                                                           value=7)

        return df_y_filtered, target_name, metric_model, metric, typ, random_state, model_list, input_driveID, rows, cols, figsize_width, figsize_height


def filter_dataframe_based_on_sidebar(df_y, trip_lengths, trip_max_accels, target_name, min_length, max_length,
                                      min_accel, max_accel, drive_ids_w_sensor_err):
    """
    Filter a dataframe based on various conditions provided through the Streamlit sidebar.

    Parameters:
    - df_y (pd.DataFrame): The input dataframe to be filtered.
    - trip_lengths (pd.DataFrame): Dataframe representing the length of trips.
    - trip_max_accels (pd.DataFrame): Dataframe representing the maximum accelerations of trips.
    - target_name (str): Name of the target column to be used in filtering.
    - min_length (int/float): Minimum trip length filter.
    - max_length (int/float): Maximum trip length filter.
    - min_accel (int/float): Minimum trip acceleration filter.
    - max_accel (int/float): Maximum trip acceleration filter.
    - drive_ids_w_sensor_err (set/list): List or set of drive IDs with sensor errors.

    Returns:
    - pd.DataFrame: Filtered dataframe based on the provided conditions.
    """

    # Filter drive IDs based on matching trip lengths
    drive_ids_with_matching_length = trip_lengths[
        (trip_lengths['timestamp_utc'] >= min_length) &
        (trip_lengths['timestamp_utc'] <= max_length)
        ].index

    # Filter drive IDs based on matching trip maximum accelerations
    drive_ids_with_matching_max_accel = trip_max_accels[
        (trip_max_accels[target_name] >= min_accel) &
        (trip_max_accels[target_name] <= max_accel)
        ].index

    # Intersect the filtered drive IDs with those having sensor errors
    filtered_drive_ids = set(drive_ids_with_matching_length) & set(drive_ids_with_matching_max_accel) & set(
        drive_ids_w_sensor_err)

    # Filter the input dataframe based on the final list of drive IDs
    df_y = df_y[df_y['driveID'].isin(list(filtered_drive_ids))]

    return df_y


@st.cache_data
def display_data_overview(data_overview):
    """
    Displays an overview of the loaded data on the Streamlit app using a given `DataOverview` instance.

    The function presents insights regarding the loaded dataset, such as number of data points,
    distinct drive IDs, mean duration for each drive, and data validity percentages. This
    information is shown on the Streamlit application with a structured markdown layout.

    Parameters:
    - data_overview (DataOverview)

    Note: The function is decorated with `@st.cache_data` to cache data retrieval operations
    and enhance display speed during repeated visits.

    Returns:
    None: The function's purpose is the side effect of rendering data on the Streamlit app.
    """

    st.title(f'Overview Of Loaded Data - {data_overview.target_name}')
    loaded_datapoints = data_overview.df_y.shape[0]
    loaded_driveIDs = len(set(data_overview.df_y['driveID']))
    loaded_average_duration = data_overview.df_y.groupby('driveID').agg({'timestamp_utc': 'count'}).mean(axis=0)/60
    loaded_validities = data_overview.df_y[['gps_valid', 'gyro_valid', 'accel_valid']].mean(axis=0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<h3 style='text-align: center; color: grey;'>Datapoints: {format(loaded_datapoints, ',')}</h3>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h3 style='text-align: center; color: grey;'>Trips: {format(loaded_driveIDs, ',')}</h3>",
                    unsafe_allow_html=True)
    with col3:
        st.markdown(
            f"<h3 style='text-align: center; color: grey;'>Average Trip Duration: {round(float(loaded_average_duration), 1)}m</h3>",
            unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<h3 style='text-align: center; color: grey;'>GPS Availibility: {round(float(loaded_validities[0]) * 100, 2)}%</h3>",
            unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"<h3 style='text-align: center; color: grey;'>Gyro and Accel Availibility: {round(float(loaded_validities[1]) * 100, 2)}%</h3>",
            unsafe_allow_html=True)


def display_metrics(df_results):
    """
    Displays the metrics on the Streamlit app.
    """

    st.title('Metrics')
    st.write(df_results)


def display_plots(sidebar_config):
    """
    Display metrics and visualizations of trips in the Streamlit app based on provided conditions.

    Parameters:
    - sidebar_config (SidebarConfig)
    Returns:
    - None: The function directly displays the results in Streamlit.
    """

    st.title('Visualization of Trips')

    if len(sidebar_config.df_y_filtered) > 0:
        plotly_figure = plot_driveIDs2(
            sidebar_config.df_y_filtered, sidebar_config.target_name, model_list=sidebar_config.model_list, d_id_list=sidebar_config.input_driveID,
            rows=sidebar_config.rows, cols=sidebar_config.cols, figsize=(sidebar_config.figsize_width, sidebar_config.figsize_height), metric=sidebar_config.metric,
            typ=sidebar_config.typ, random_state=sidebar_config.random_state, save=False,
            path=f'/home/stud03/data_science_challenge/Louis/files_huk/plots/{sidebar_config.target_name}_meta_on_invalids_20.png',
            metric_model=sidebar_config.metric_model
        )

        st.plotly_chart(plotly_figure)  # Display the visualization
    else:
        st.warning('No matching data for the specified criteria.')




