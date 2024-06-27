import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Constants
DESIRED_LOCATIONS = ['Pacers 14th St', 'Pacers Georgetown', 'Pacers Alexandria', 'Pacers Navy Yard', 'Pacers Arlington']
FILTERED_PRODUCT_TYPES = ["Men's Shoes", "Women's Shoes", "Footbeds", "Socks"]
START_DATE = pd.to_datetime('2024-04-01')

# Set the page configuration
st.set_page_config(
    page_title="Quarterly Report",
    page_icon=":smile:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def clean_data(data):
    """Clean the input data by removing duplicates and handling missing values."""
    data = data.drop_duplicates()
    data = data.dropna(subset=['pos_location_name'])
    data = data[data['pos_location_name'].isin(DESIRED_LOCATIONS)]
    data['staff_name'] = data['staff_name'].fillna(data['pos_location_name'])
    data['name_of_staff_who_helped_with_sale'] = data['name_of_staff_who_helped_with_sale'].fillna(data['staff_name'])

    # Standardize the capitalization of staff names
    data['name_of_staff_who_helped_with_sale'] = data['name_of_staff_who_helped_with_sale'].str.title()

    data = data.dropna(subset=['product_type', 'variant_title'])

    return data

def add_week_column(data, date_column, week_column):
    """Add a week column based on the specified date column."""
    data.loc[:, date_column] = pd.to_datetime(data[date_column], errors='coerce').dt.date
    data = data.dropna(subset=[date_column])
    data.loc[:, week_column] = pd.to_datetime(data[date_column], errors='coerce').dt.to_period('W').apply(lambda r: r.start_time)
    return data

def filter_data_by_product_type(data, product_types):
    """Filter the data by specified product types."""
    return data[data['product_type'].isin(product_types)]

def aggregate_data_by_columns(data, group_columns, agg_column):
    """Aggregate the data by specified columns using pivot table."""
    pivot_table = pd.pivot_table(data, values=agg_column, index=group_columns, columns='product_type', aggfunc='sum', fill_value=0)
    pivot_table.reset_index(inplace=True)
    pivot_table.columns = ['_'.join(col).strip() if type(col) is tuple else col for col in pivot_table.columns.values]
    return pivot_table

def rename_columns(data):
    """Rename columns to appropriate names."""
    data.rename(columns={
        "Men's Shoes": 'net_quantity_mens_shoes',
        "Women's Shoes": 'net_quantity_womens_shoes',
        'Socks': 'net_quantity_socks',
        'Footbeds': 'net_quantity_insoles'
    }, inplace=True)
    return data

def combine_shoe_columns(data):
    """Combine men's and women's shoes into a single column."""
    data['net_quantity_shoes'] = data['net_quantity_mens_shoes'] + data['net_quantity_womens_shoes']
    return data

def fill_and_convert(data):
    """Fill NaN values and convert data types."""
    data['net_quantity_socks'] = data['net_quantity_socks'].fillna(0)
    data['net_quantity_insoles'] = data['net_quantity_insoles'].fillna(0)
    data['net_quantity_shoes'] = data['net_quantity_shoes'].astype(int)
    data['net_quantity_socks'] = data['net_quantity_socks'].astype(int)
    data['net_quantity_insoles'] = data['net_quantity_insoles'].astype(int)
    return data

def calculate_ratios(data):
    """Calculate CCI ratios."""
    # Calculate the ratios
    data['sock_to_shoe_ratio'] = data['net_quantity_socks'] / data['net_quantity_shoes']
    data['insole_to_shoe_ratio'] = data['net_quantity_insoles'] / data['net_quantity_shoes']

    # Replace null and infinite values after the ratio calculation
    data['sock_to_shoe_ratio'] = data['sock_to_shoe_ratio'].replace([float('inf'), -float('inf')], None)
    data['insole_to_shoe_ratio'] = data['insole_to_shoe_ratio'].replace([float('inf'), -float('inf')], None)

    # Ensure the columns are of numeric type before filling NA values
    data['sock_to_shoe_ratio'] = pd.to_numeric(data['sock_to_shoe_ratio'], errors='coerce')
    data['insole_to_shoe_ratio'] = pd.to_numeric(data['insole_to_shoe_ratio'], errors='coerce')

    data['sock_to_shoe_ratio'] = data['sock_to_shoe_ratio'].fillna(0)
    data['insole_to_shoe_ratio'] = data['insole_to_shoe_ratio'].fillna(0)

    # Multiply by 100 and round to 2 decimal places
    # data['sock_to_shoe_ratio'] = (data['sock_to_shoe_ratio'] * 100).round(2)
    # data['insole_to_shoe_ratio'] = (data['insole_to_shoe_ratio'] * 100).round(2)

    data['sock_to_shoe_ratio'] = (data['sock_to_shoe_ratio']).round(2)
    data['insole_to_shoe_ratio'] = (data['insole_to_shoe_ratio']).round(2)

    return data

def process_week_data(data, week_column):
    """Convert week dates to week numbers and add date ranges."""
    # Convert week dates to week numbers
    data[week_column] = ((data[week_column] - START_DATE) / pd.Timedelta(weeks=1)).astype(int) + 1

    # Add date ranges
    data['date_range'] = data[week_column].apply(
        lambda
            x: f"{(START_DATE + pd.Timedelta(days=(x - 1) * 7)).strftime('%b %d')} - {(START_DATE + pd.Timedelta(days=x * 7 - 1)).strftime('%b %d')}"
    )

    return data

def calculate_total_quantities(data):
    """Calculate total quantities and ratios."""
    total_quantities = data[['net_quantity_shoes', 'net_quantity_socks', 'net_quantity_insoles']].sum()
    total_quantities['sock_to_shoe_ratio'] = total_quantities['net_quantity_socks'] / total_quantities['net_quantity_shoes']
    total_quantities['insole_to_shoe_ratio'] = total_quantities['net_quantity_insoles'] / total_quantities['net_quantity_shoes']
    total_quantities_df = total_quantities.to_frame().T
    return total_quantities_df

#####
# Quarter Data
#####

def calculate_quarter_totals(data):
    """Calculate quarter totals and ratios."""
    quarter_total = data.groupby('pos_location_name')[['net_quantity_shoes', 'net_quantity_socks', 'net_quantity_insoles']].sum().reset_index()
    quarter_total['sock_to_shoe_ratio'] = quarter_total['net_quantity_socks'] / quarter_total['net_quantity_shoes']
    quarter_total['insole_to_shoe_ratio'] = quarter_total['net_quantity_insoles'] / quarter_total['net_quantity_shoes']
    return quarter_total

def calculate_weekly_totals(data):
    """Calculate weekly totals and ratios."""
    weekly_total = data.groupby(['week', 'pos_location_name'])[['net_quantity_shoes', 'net_quantity_socks', 'net_quantity_insoles']].sum().reset_index()
    weekly_total['sock_to_shoe_ratio'] = weekly_total['net_quantity_socks'] / weekly_total['net_quantity_shoes']
    weekly_total['insole_to_shoe_ratio'] = weekly_total['net_quantity_insoles'] / weekly_total['net_quantity_shoes']
    return weekly_total



def create_quarter_charts(quarter_total):
    """Create and display quarter charts."""
    total_shoes = quarter_total['net_quantity_shoes'].sum()
    total_socks = quarter_total['net_quantity_socks'].sum()
    total_insoles = quarter_total['net_quantity_insoles'].sum()
    # total_sock_ratio = quarter_total['sock_to_shoe_ratio'].mean()
    total_sock_ratio = total_socks/total_shoes
    total_insole_ratio = quarter_total['insole_to_shoe_ratio'].mean()

    stores = quarter_total['pos_location_name'].unique()
    categories = ['Shoes', 'Socks', 'Insoles']
    column_mapping = ['net_quantity_shoes', 'net_quantity_socks', 'net_quantity_insoles']

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = create_cci_bar_chart_grouped(quarter_total, categories, stores, column_mapping)
        st.pyplot(fig)

    with col2:
        fig = create_donut_chart(f"{total_sock_ratio:.2%}", quarter_total['sock_to_shoe_ratio'],
                                 quarter_total['pos_location_name'], 'Sock to Shoe Ratio for Quarter')
        st.pyplot(fig)

    with col3:
        fig = create_donut_chart(f"{total_insole_ratio:.2%}", quarter_total['insole_to_shoe_ratio'],
                                 quarter_total['pos_location_name'], 'Insole to Shoe Ratio for Quarter')
        st.pyplot(fig)

def create_weekly_charts(weekly_total):
    """Create and display weekly charts."""
    total_shoes = weekly_total['net_quantity_shoes'].sum()
    total_socks = weekly_total['net_quantity_socks'].sum()
    total_insoles = weekly_total['net_quantity_insoles'].sum()
    total_sock_ratio = total_socks / total_shoes
    total_insole_ratio = total_insoles / total_shoes

    stores = weekly_total['pos_location_name'].unique()
    categories = ['Shoes', 'Socks', 'Insoles']
    column_mapping = ['net_quantity_shoes', 'net_quantity_socks', 'net_quantity_insoles']

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = create_cci_bar_chart_grouped(weekly_total, categories, stores, column_mapping)
        st.pyplot(fig)

    with col2:
        fig = create_donut_chart(f"{total_sock_ratio:.2%}", weekly_total['sock_to_shoe_ratio'],
                                 weekly_total['pos_location_name'], 'Sock to Shoe Ratio for Week')
        st.pyplot(fig)

    with col3:
        fig = create_donut_chart(f"{total_insole_ratio:.2%}", weekly_total['insole_to_shoe_ratio'],
                                 weekly_total['pos_location_name'], 'Insole to Shoe Ratio for Week')
        st.pyplot(fig)

def create_cci_bar_chart_grouped(data, categories, stores, column_mapping):
    """Create a grouped bar chart for total quantities per store."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract values for each store and category
    store_values = {}
    for store in stores:
        store_values[store] = [data[data['pos_location_name'] == store][col].sum() for col in column_mapping]

    # Determine the number of categories and stores
    n_categories = len(categories)
    n_stores = len(stores)

    # Create bar positions
    bar_width = 0.15
    bar_positions = [i + bar_width * np.arange(n_stores) for i in range(n_categories)]

    # Plot bars for each store
    for idx, store in enumerate(stores):
        values = store_values[store]
        ax.bar([pos + idx * bar_width for pos in range(n_categories)], values, bar_width, label=store, edgecolor='black')

    ax.set_ylabel('Total Quantities')
    ax.set_title('Total Quantities for Quarter by Store', fontweight='bold')
    ax.set_xticks([r + bar_width * (n_stores - 1) / 2 for r in range(n_categories)])
    ax.set_xticklabels(categories)
    ax.legend()

    # Add text labels on bars
    for idx, store in enumerate(stores):
        values = store_values[store]
        for i, v in enumerate(values):
            ax.text(i + idx * bar_width, v + 3, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')

    return fig

def wrap_labels(labels):
    wrapped_labels = []
    for label in labels:
        if ' ' in label:
            parts = label.split(' ', 1)
            wrapped_label = parts[0] + '\n' + parts[1]
        else:
            wrapped_label = label
        wrapped_labels.append(wrapped_label)
    return wrapped_labels

def create_donut_chart(total, values, labels, title):
    fig, ax = plt.subplots()

    # Wrap labels to place "Pacers" on the top line and the rest on the bottom line
    wrapped_labels = wrap_labels(labels)

    wedges, texts, autotexts = ax.pie(values, labels=wrapped_labels, autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '', startangle=90,
                                      wedgeprops=dict(width=0.2, edgecolor='black'), pctdistance=0.65)

    # Correct the autopct to display correct percentages
    for i, a in enumerate(autotexts):
        a.set_text(f'{values[i] * 100:.1f}%')

    # Draw white circle in the center to create donut effect
    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(center_circle)

    # Add total number in the center
    plt.text(0, 0, str(total), ha='center', va='center', fontsize=26, color='black')

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    plt.title(title, fontweight='bold')
    return fig

#####
#Individual staff data
#####

def prepare_store_dataframes(data):
    """Prepare dataframes for each store and sort by staff names."""
    stores = ['Pacers 14th St', 'Pacers Georgetown', 'Pacers Alexandria', 'Pacers Navy Yard', 'Pacers Arlington']
    store_dataframes = {}
    for store in stores:
        store_df = data[data['pos_location_name'] == store]
        store_df_sorted = store_df.sort_values(by='name_of_staff_who_helped_with_sale')
        store_dataframes[store] = store_df_sorted
    return store_dataframes

def handle_calculations(df):
    """Prepare the data by calculating ratios and handling infinite values."""
    df = df.copy()  # Ensure we are working on a copy of the DataFrame
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['insole_to_shoe_ratio'] = df['net_quantity_insoles'] / df['net_quantity_shoes']
    df['sock_to_shoe_ratio'] = df['net_quantity_socks'] / df['net_quantity_shoes']
    return df

def create_staff_charts(store_dataframes):
    """Create and display weekly charts."""
    store_options = list(store_dataframes.keys())
    store_name = st.selectbox("Select a store:", store_options)

    if store_name:
        store_df = store_dataframes[store_name]
        store_df = handle_calculations(store_df)
        staff_members = store_df['name_of_staff_who_helped_with_sale'].unique()

        staff_name = st.selectbox("Select a staff member:", staff_members)

        if staff_name:
            staff_data = store_df[store_df['name_of_staff_who_helped_with_sale'] == staff_name]

            col1, col2 = st.columns(2)

            with col1:
                create_table(staff_data)
            with col2:
                create_staff_bar_chart(staff_data)

def create_table(staff_data):
    """Create a table for staff data."""
    # Select relevant columns and copy data
    table_data = staff_data[
        ['week', 'date_range', 'net_quantity_shoes', 'net_quantity_socks', 'net_quantity_insoles']].copy()

    # Rename columns
    table_data.columns = ['Week', 'Date Range', 'Shoes Sold', 'Socks Sold', 'Insoles Sold']

    # Sort data by 'Week' in ascending order
    table_data.sort_values('Week', inplace=True)

    # Set 'Week' as index
    table_data.set_index('Week', inplace=True)

    # Display table
    st.table(table_data)

def create_staff_bar_chart(staff_data):
    """Create a bar chart for staff data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    spacing = 0.15

    bars_sock = ax.bar(staff_data['week'] - (bar_width / 2 + spacing / 2), staff_data['sock_to_shoe_ratio'],
                       width=bar_width, label='Sock-to-Shoe Ratio', color='skyblue', edgecolor='black')
    bars_insole = ax.bar(staff_data['week'] + (bar_width / 2 + spacing / 2), staff_data['insole_to_shoe_ratio'],
                         width=bar_width, label='Insole-to-Shoe Ratio', color='red', edgecolor='black')

    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.tick_params(width=1.5)

    for bar in bars_sock:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.0%}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    for bar in bars_insole:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.0%}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    ax.set_xlabel('Week Number')
    ax.set_ylabel('Ratio')
    ax.set_title('Ratios Over Time')

    max_sock_ratio = staff_data['sock_to_shoe_ratio'].replace([np.inf, -np.inf], np.nan).dropna().max()
    max_insole_ratio = staff_data['insole_to_shoe_ratio'].replace([np.inf, -np.inf], np.nan).dropna().max()
    max_ratio = max(max_sock_ratio, max_insole_ratio)

    if pd.notnull(max_ratio) and max_ratio > 0:
        ax.set_ylim(0, max_ratio * 1.2)
    else:
        ax.set_ylim(0, 1)

    ax.set_xticks(staff_data['week'])
    ax.set_xticklabels(staff_data['week'])
    ax.legend()

    st.pyplot(fig)

# Streamlit title
st.header("Q2 CCI Dashboard")

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload Shopify Report", type=["csv"])

if uploaded_file is not None:
    try:

        # Upload data file
        data = pd.read_csv(uploaded_file)

        # Clean data
        cleaned_data = clean_data(data)

        # Filter data by products of interest
        cci_sales = cleaned_data[cleaned_data['product_type'].isin(FILTERED_PRODUCT_TYPES)]

        # Add a column 'week' to dataframe
        cci_sales = add_week_column(cci_sales, 'day', 'week')

        # Filter data into specific dataframes by product
        shoes_df = filter_data_by_product_type(cci_sales, ["Men's Shoes", "Women's Shoes"])
        socks_df = filter_data_by_product_type(cci_sales, ["Socks"])
        insoles_df = filter_data_by_product_type(cci_sales, ["Footbeds"])

        group_columns = ['week', 'pos_location_name', 'name_of_staff_who_helped_with_sale']

        # Aggregate data
        agg_data1 = aggregate_data_by_columns(cci_sales, group_columns, 'net_quantity')

        # Rename columns
        agg_data2 = rename_columns(agg_data1)

        # Combine men's and women's shoes
        agg_data3 = combine_shoe_columns(agg_data2)

        # Fill NaN values and convert data types
        agg_data4 = fill_and_convert(agg_data3)

        # Calculate CCI ratios
        agg_ratio_df = calculate_ratios(agg_data4)

        # Process the 'week' column and add date ranges
        store_data_clean_df = process_week_data(agg_ratio_df, 'week')

        # Calculate quarter data
        st.header("Quarter Data")

        quarter_total = calculate_quarter_totals(store_data_clean_df)
        total_quantities_df = calculate_total_quantities(store_data_clean_df)

        # Display quarter charts
        create_quarter_charts(quarter_total)

        # Weekly data section
        st.header("Weekly Data")

        # Group by week and pos_location_name and sum the net_quantity columns
        weekly_totals_by_location = store_data_clean_df.groupby(['week', 'date_range', 'pos_location_name'])[
            ['net_quantity_shoes', 'net_quantity_socks', 'net_quantity_insoles']
        ].sum().reset_index()

        # Calculate ratios
        weekly_totals_by_location = calculate_ratios(weekly_totals_by_location)

        # Sort the date ranges inline without adding a new column or function
        week_range = sorted(weekly_totals_by_location['date_range'].unique(),
                            key=lambda x: (pd.to_datetime(x.split(' - ')[0], format='%b %d')),
                            reverse=True)

        # Week selection
        selected_week = st.selectbox("Select a Week", week_range)

        # Filter the DataFrame for the selected week
        specific_week_totals = weekly_totals_by_location[weekly_totals_by_location['date_range'] == selected_week].drop(columns=['week', 'date_range']).reset_index(drop=True)

        create_weekly_charts(specific_week_totals)

        # Display Staff Data
        st.header("Staff Trend Data")

        # Prepare store dataframes and display store level and individual staff charts
        store_dataframes = prepare_store_dataframes(store_data_clean_df)
        create_staff_charts(store_dataframes)

    except Exception as e:
        st.error(f"Error reading the file. Please ensure you are using the correctly formatted Shopify Report. Contact Ivan for assistance.")
