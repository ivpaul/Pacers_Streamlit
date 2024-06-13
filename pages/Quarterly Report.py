import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the page configuration
# st.set_page_config(
#     page_title="Quarterly Report",
#     page_icon="🧊",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

def prepare_data(df):
    # Ensure the insole-to-shoe ratio is calculated and handle infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if 'insole_to_shoe_ratio' not in df.columns:
        df['insole_to_shoe_ratio'] = df['net_quantity_insoles'] / df['net_quantity_shoes']
    if 'sock_to_shoe_ratio' not in df.columns:
        df['sock_to_shoe_ratio'] = df['net_quantity_socks'] / df['net_quantity_shoes']
    return df

def add_date_ranges(df):
    start_date = pd.to_datetime('2024-04-01')
    df['date_range'] = df['week'].apply(lambda x: f"{(start_date + pd.Timedelta(days=(x-1)*7)).strftime('%Y-%m-%d')} to {(start_date + pd.Timedelta(days=x*7 - 1)).strftime('%Y-%m-%d')}")
    return df

def create_bar_chart(staff_data):
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

def create_table(staff_data):
    table_data = staff_data[
        ['week', 'date_range', 'net_quantity_shoes', 'net_quantity_socks', 'net_quantity_insoles']].copy()
    table_data.columns = ['Week', 'Date Range', 'Shoes Sold', 'Socks Sold', 'Insoles Sold']
    table_data.set_index('Week', inplace=True)

    st.table(table_data)


# Streamlit title
st.title("Q2 Individual Staff CCI Data")

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file

    data = pd.read_csv(uploaded_file)

    sales_data = data[['day', 'staff_name', 'name_of_staff_who_helped_with_sale', 'product_type', 'net_quantity',
                           'pos_location_name']]
    sales_data_clean = sales_data.dropna(subset=['pos_location_name'])

    filtered_product_types = ["Men's Shoes", "Women's Shoes", "Unisex Shoes", "Footbeds", "Socks"]
    cci_sales = sales_data_clean[sales_data_clean['product_type'].isin(filtered_product_types)]

    # Convert the day column to datetime format
    cci_sales['day'] = pd.to_datetime(cci_sales['day'])
    # Add a 'week' column to the DataFrame
    cci_sales['week'] = cci_sales['day'].dt.to_period('W').apply(lambda r: r.start_time)

    # Calculate totals for shoes and socks
    shoes_df = cci_sales[cci_sales['product_type'].isin(["Men's Shoes", "Women's Shoes", "Unisex Shoes"])]
    socks_df = cci_sales[cci_sales['product_type'] == "Socks"]
    insoles_df = cci_sales[cci_sales['product_type'] == "Footbeds"]

    # Group by week and staff, then sum net_quantity for shoes and socks
    shoes_agg = shoes_df.groupby(['week', 'name_of_staff_who_helped_with_sale', 'pos_location_name'])[
        'net_quantity'].sum().reset_index()
    socks_agg = socks_df.groupby(['week', 'name_of_staff_who_helped_with_sale', 'pos_location_name'])[
        'net_quantity'].sum().reset_index()
    insoles_agg = insoles_df.groupby(['week', 'name_of_staff_who_helped_with_sale', 'pos_location_name'])[
        'net_quantity'].sum().reset_index()

    merged_agg = pd.merge(shoes_agg, socks_agg, on=['week', 'name_of_staff_who_helped_with_sale', 'pos_location_name'],
                          how='left', suffixes=('_shoes', '_socks'))
    merged_agg = pd.merge(merged_agg, insoles_agg,
                          on=['week', 'name_of_staff_who_helped_with_sale', 'pos_location_name'], how='left')

    # Rename the columns for clarity
    merged_agg.rename(columns={'net_quantity': 'net_quantity_insoles'}, inplace=True)
    merged_agg['net_quantity_socks'].fillna(0, inplace=True)
    merged_agg['net_quantity_insoles'].fillna(0, inplace=True)

    # Convert net_quantity_socks and net_quantity_shoes to integers
    merged_agg['net_quantity_shoes'] = merged_agg['net_quantity_shoes'].astype(int)
    merged_agg['net_quantity_socks'] = merged_agg['net_quantity_socks'].astype(int)
    merged_agg['net_quantity_insoles'] = merged_agg['net_quantity_insoles'].astype(int)

    # Calculate the sock-to-shoe ratio
    merged_agg['sock_to_shoe_ratio'] = merged_agg['net_quantity_socks'] / merged_agg['net_quantity_shoes']
    merged_agg['insole_to_shoe_ratio'] = merged_agg['net_quantity_insoles'] / merged_agg['net_quantity_shoes']

    start_date = pd.to_datetime('2024-04-01')

    # Calculate the week number relative to the start date
    merged_agg['week'] = ((merged_agg['week'] - start_date) / pd.Timedelta(weeks=1)).astype(int) + 1
    merged_agg = merged_agg.sort_values(by=['name_of_staff_who_helped_with_sale', 'week'])

    store_data_clean_df = add_date_ranges(merged_agg)

    # Assuming the CSV file has the necessary columns including 'pos_location_name'
    Q2_cci_14_clean_df = store_data_clean_df[store_data_clean_df['pos_location_name'] == 'Pacers 14th St']
    Q2_cci_gt_clean_df = store_data_clean_df[store_data_clean_df['pos_location_name'] == 'Pacers Georgetown']
    Q2_cci_ot_clean_df = store_data_clean_df[store_data_clean_df['pos_location_name'] == 'Pacers Alexandria']
    Q2_cci_ny_clean_df = store_data_clean_df[store_data_clean_df['pos_location_name'] == 'Pacers Navy Yard']
    Q2_cci_arl_clean_df = store_data_clean_df[store_data_clean_df['pos_location_name'] == 'Pacers Arlington']

    # Prepare the data
    store_dataframes = {
        'Pacers 14th St': Q2_cci_14_clean_df,
        'Pacers Georgetown': Q2_cci_gt_clean_df,
        'Pacers Alexandria': Q2_cci_ot_clean_df,
        'Pacers Navy Yard': Q2_cci_ny_clean_df,
        'Pacers Arlington': Q2_cci_arl_clean_df
    }

    # Selectbox for store selection
    store_options = list(store_dataframes.keys())
    store_name = st.selectbox("Select the store:", store_options)

    if store_name:
        st.subheader(f'Reports for {store_name}')
        store_df = store_dataframes[store_name]
        store_df = prepare_data(store_df)  # Ensure data is prepared after selection
        staff_members = store_df['name_of_staff_who_helped_with_sale'].unique()

        # Selectbox for staff member selection
        staff_name = st.selectbox("Select a staff member:", staff_members)

        if staff_name:
            st.write(f'##### Report for {staff_name}')
            staff_data = store_df[store_df['name_of_staff_who_helped_with_sale'] == staff_name]
            create_table(staff_data)
            create_bar_chart(staff_data)
else:
    st.info("Please upload a CSV file to proceed.")
