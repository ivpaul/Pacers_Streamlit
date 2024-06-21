import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

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


def log_access(message):
    """Log access to the application."""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("access_log.txt", "a") as f:
        f.write(f"{message} at: {current_time}\n")

def clean_data(data):
    """Clean the input data by removing duplicates and handling missing values."""
    data = data.drop_duplicates()
    data = data.dropna(subset=['pos_location_name'])

    data = data[data['pos_location_name'].isin(DESIRED_LOCATIONS)]

    data['staff_name'] = data['staff_name'].fillna(data['pos_location_name'])
    data['name_of_staff_who_helped_with_sale'] = data['name_of_staff_who_helped_with_sale'].fillna(data['staff_name'])
    data = data.dropna(subset=['product_type', 'variant_title'])

    return data

def filter_data_by_product_type(data, product_types):
    """Filter the data by specified product types."""
    return data[data['product_type'].isin(product_types)]

def aggregate_data_by_columns(data, group_columns, agg_column):
    """Aggregate the data by specified columns."""
    return data.groupby(group_columns)[agg_column].sum().reset_index()

def merge_aggregated_data(shoes_agg, socks_agg, insoles_agg):
    """Merge aggregated data for shoes, socks, and insoles."""
    merged_agg = pd.merge(shoes_agg, socks_agg, on=['week', 'pos_location_name', 'name_of_staff_who_helped_with_sale'],
                          how='left', suffixes=('_shoes', '_socks'))
    merged_agg = pd.merge(merged_agg, insoles_agg,
                          on=['week', 'pos_location_name', 'name_of_staff_who_helped_with_sale'], how='left')
    return merged_agg

def prepare_data(df):
    """Prepare the data by calculating ratios and handling infinite values."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['insole_to_shoe_ratio'] = df['net_quantity_insoles'] / df['net_quantity_shoes']
    df['sock_to_shoe_ratio'] = df['net_quantity_socks'] / df['net_quantity_shoes']
    return df

def add_date_ranges(df):
    """Add date ranges to the DataFrame."""
    df['date_range'] = df['week'].apply(lambda
                                            x: f"{(START_DATE + pd.Timedelta(days=(x - 1) * 7)).strftime('%Y-%m-%d')} to {(START_DATE + pd.Timedelta(days=x * 7 - 1)).strftime('%Y-%m-%d')}")
    return df

def create_bar_chart(staff_data):
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

def create_bar_chart1(shoes, socks, insoles):
    """Create a bar chart for total quantities."""
    fig, ax = plt.subplots()
    categories = ['Shoes', 'Socks', 'Insoles']
    values = [shoes, socks, insoles]

    ax.bar(categories, values, color=['blue', 'orange', 'green'])
    ax.set_ylabel('Total Quantities')
    ax.set_title('Total Quantities for Quarter', fontweight='bold')

    for i, v in enumerate(values):
        ax.text(i, v + 3, str(v), ha='center', va='bottom', fontsize=12)

    return fig

def create_table(staff_data):
    """Create a table for staff data."""
    table_data = staff_data[
        ['week', 'date_range', 'net_quantity_shoes', 'net_quantity_socks', 'net_quantity_insoles']].copy()
    table_data.columns = ['Week', 'Date Range', 'Shoes Sold', 'Socks Sold', 'Insoles Sold']
    table_data.set_index('Week', inplace=True)

    st.table(table_data)

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

    # Create pie chart with adjusted distance for percentages
    wedges, texts, autotexts = ax.pie(values, labels=wrapped_labels, autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '', startangle=90,
                                      wedgeprops=dict(width=0.2, edgecolor='w'), pctdistance=0.65)

    # Draw white circle in the center to create donut effect
    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(center_circle)

    # Add total number in the center
    plt.text(0, 0, str(total), ha='center', va='center', fontsize=26, color='black')

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    plt.title(title, fontweight='bold')
    return fig

# Streamlit title
st.header("Q2 CCI Dashboard")

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:

    # Log the file upload event
    log_access(f"Quarterly report uploaded: {uploaded_file.name}")

    data = pd.read_csv(uploaded_file)

    # Call the clean_data function
    cleaned_data = clean_data(data)

    cci_sales = cleaned_data[cleaned_data['product_type'].isin(FILTERED_PRODUCT_TYPES)]

    cci_sales['day'] = pd.to_datetime(cci_sales['day'])
    cci_sales['week'] = cci_sales['day'].dt.to_period('W').apply(lambda r: r.start_time)

    shoes_df = filter_data_by_product_type(cci_sales, ["Men's Shoes", "Women's Shoes"])
    socks_df = filter_data_by_product_type(cci_sales, ["Socks"])
    insoles_df = filter_data_by_product_type(cci_sales, ["Footbeds"])

    group_columns = ['week', 'pos_location_name', 'name_of_staff_who_helped_with_sale']
    shoes_agg = aggregate_data_by_columns(shoes_df, group_columns, 'net_quantity')
    socks_agg = aggregate_data_by_columns(socks_df, group_columns, 'net_quantity')
    insoles_agg = aggregate_data_by_columns(insoles_df, group_columns, 'net_quantity')

    merged_agg = merge_aggregated_data(shoes_agg, socks_agg, insoles_agg)
    merged_agg.rename(columns={'net_quantity': 'net_quantity_insoles'}, inplace=True)
    merged_agg['net_quantity_socks'].fillna(0, inplace=True)
    merged_agg['net_quantity_insoles'].fillna(0, inplace=True)

    merged_agg['net_quantity_shoes'] = merged_agg['net_quantity_shoes'].astype(int)
    merged_agg['net_quantity_socks'] = merged_agg['net_quantity_socks'].astype(int)
    merged_agg['net_quantity_insoles'] = merged_agg['net_quantity_insoles'].astype(int)

    merged_agg['sock_to_shoe_ratio'] = merged_agg['net_quantity_socks'] / merged_agg['net_quantity_shoes']
    merged_agg['insole_to_shoe_ratio'] = merged_agg['net_quantity_insoles'] / merged_agg['net_quantity_shoes']

    merged_agg['week'] = ((merged_agg['week'] - START_DATE) / pd.Timedelta(weeks=1)).astype(int) + 1
    merged_agg = merged_agg.sort_values(by=['name_of_staff_who_helped_with_sale', 'week'])

    store_data_clean_df = add_date_ranges(merged_agg)

    quarter_total = store_data_clean_df.groupby('pos_location_name')[
        ['net_quantity_shoes', 'net_quantity_socks', 'net_quantity_insoles']].sum().reset_index()
    quarter_total['sock_to_shoe_ratio'] = quarter_total['net_quantity_socks'] / quarter_total['net_quantity_shoes']
    quarter_total['insole_to_shoe_ratio'] = quarter_total['net_quantity_insoles'] / quarter_total['net_quantity_shoes']

    total_quantities = store_data_clean_df[['net_quantity_shoes', 'net_quantity_socks', 'net_quantity_insoles']].sum()
    total_quantities['sock_to_shoe_ratio'] = total_quantities['net_quantity_socks'] / total_quantities[
        'net_quantity_shoes']
    total_quantities['insole_to_shoe_ratio'] = total_quantities['net_quantity_insoles'] / total_quantities[
        'net_quantity_shoes']
    total_quantities_df = total_quantities.to_frame().T

    total_shoes = quarter_total['net_quantity_shoes'].sum()
    total_socks = quarter_total['net_quantity_socks'].sum()
    total_insoles = quarter_total['net_quantity_insoles'].sum()
    total_sock_ratio = quarter_total['sock_to_shoe_ratio'].mean()
    total_insole_ratio = quarter_total['insole_to_shoe_ratio'].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = create_bar_chart1(total_shoes, total_socks, total_insoles)
        st.pyplot(fig)

    with col2:
        fig = create_donut_chart(f"{total_sock_ratio:.2%}", quarter_total['sock_to_shoe_ratio'],
                                 quarter_total['pos_location_name'], 'Sock to Shoe Ratio for Quarter')
        st.pyplot(fig)

    with col3:
        fig = create_donut_chart(f"{total_insole_ratio:.2%}", quarter_total['insole_to_shoe_ratio'],
                                 quarter_total['pos_location_name'], 'Insole to Shoe Ratio for Quarter')
        st.pyplot(fig)

    Q2_cci_14_clean_df = store_data_clean_df[store_data_clean_df['pos_location_name'] == 'Pacers 14th St']
    Q2_cci_gt_clean_df = store_data_clean_df[store_data_clean_df['pos_location_name'] == 'Pacers Georgetown']
    Q2_cci_ot_clean_df = store_data_clean_df[store_data_clean_df['pos_location_name'] == 'Pacers Alexandria']
    Q2_cci_ny_clean_df = store_data_clean_df[store_data_clean_df['pos_location_name'] == 'Pacers Navy Yard']
    Q2_cci_arl_clean_df = store_data_clean_df[store_data_clean_df['pos_location_name'] == 'Pacers Arlington']

    store_dataframes = {
        'Pacers 14th St': Q2_cci_14_clean_df,
        'Pacers Georgetown': Q2_cci_gt_clean_df,
        'Pacers Alexandria': Q2_cci_ot_clean_df,
        'Pacers Navy Yard': Q2_cci_ny_clean_df,
        'Pacers Arlington': Q2_cci_arl_clean_df
    }

    store_options = list(store_dataframes.keys())
    store_name = st.selectbox("Select a store:", store_options)

    if store_name:
        store_df = store_dataframes[store_name]
        store_df = prepare_data(store_df)
        staff_members = store_df['name_of_staff_who_helped_with_sale'].unique()

        staff_name = st.selectbox("Select a staff member:", staff_members)

        if staff_name:
            staff_data = store_df[store_df['name_of_staff_who_helped_with_sale'] == staff_name]

            col1, col2 = st.columns(2)

            with col1:
                create_table(staff_data)
            with col2:
                create_bar_chart(staff_data)
else:
    st.info("Please upload a CSV file to proceed.")
