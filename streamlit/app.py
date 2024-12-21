
from streamlit_utils import display_plots, load_and_process_data, display_data_overview, display_metrics, \
    configure_layout, DataOverview, DataAnalysis, SidebarConfig, analyze_data, sidebar_configuration


if __name__ == '__main__':
    uploaded_file = configure_layout()

    if uploaded_file is not None:
        # Data Preparation
        df_y, model_list_input, targets_available, df_results = load_and_process_data(uploaded_file)

        # Get Data Overview
        data_overview = DataOverview(df_y, targets_available[0])

        # Display Data Overview
        display_data_overview(data_overview)

        # Display Metrics
        display_metrics(df_results)

        # Data Analysis
        data_analysis = DataAnalysis(*analyze_data(data_overview.df_y, data_overview.target_name))

        # Get Sidebar Configuration
        sidebar_config = SidebarConfig(*sidebar_configuration(data_overview, targets_available, model_list_input, data_analysis))

        # Display Plots
        display_plots(sidebar_config)
