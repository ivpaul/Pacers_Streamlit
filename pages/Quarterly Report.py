import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the page configuration
st.set_page_config(
    page_title="Quarterly Report",
    page_icon=":smile:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def clean_data(data):
    # Data cleaning steps
    data = data.drop_duplicates()
    data = data.dropna(subset=['pos_location_name'])

    # List of desired locations
    desired_locations = ['Pacers 14th St',
                         'Pacers Georgetown',
                         'Pacers Alexandria',
                         'Pacers Navy Yard',
                         'Pacers Arlington']

    # Filter the DataFrame to include only the desired locations
    data = data[data['pos_location_name'].isin(desired_locations)]

    data['staff_name'] = data['staff_name'].fillna(data['pos_location_name'])
    data['name_of_staff_who_helped_with_sale'] = data['name_of_staff_who_helped_with_sale'].fillna(data['staff_name'])
    data = data.dropna(subset=['product_type'])
    data = data.dropna(subset=['variant_title'])

    return data

def filter_data_by_product_type(data, product_types):
    return data[data['product_type'].isin(product_types)]

def aggregate_data_by_columns(data, group_columns, agg_column):
    return data.groupby(group_columns)[agg_column].sum().reset_index()

def merge_aggregated_data(shoes_agg, socks_agg, insoles_agg):
    merged_agg = pd.merge(shoes_agg, socks_agg, on=['week', 'pos_location_name', 'name_of_staff_who_helped_with_sale'],
                          how='left', suffixes=('_shoes', '_socks'))
    merged_agg = pd.merge(merged_agg, insoles_agg,
                          on=['week', 'pos_location_name', 'name_of_staff_who_helped_with_sale'], how='left')
    return merged_agg


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

def create_bar_chart1(shoes, socks, insoles):
    fig, ax = plt.subplots()
    categories = ['Shoes', 'Socks', 'Insoles']
    values = [shoes, socks, insoles]

    ax.bar(categories, values, color=['blue', 'orange', 'green'])
    ax.set_ylabel('Total Quantities')
    ax.set_title('Total Quantities for Quarter')

    for i, v in enumerate(values):
        ax.text(i, v + 3, str(v), ha='center', va='bottom', fontsize=12)

    return fig

def create_table(staff_data):
    table_data = staff_data[
        ['week', 'date_range', 'net_quantity_shoes', 'net_quantity_socks', 'net_quantity_insoles']].copy()
    table_data.columns = ['Week', 'Date Range', 'Shoes Sold', 'Socks Sold', 'Insoles Sold']
    table_data.set_index('Week', inplace=True)

    st.table(table_data)

def create_donut_chart(total, values, labels, title, is_percentage=False):
    fig, ax = plt.subplots()

    # Convert values to percentages for display
    percent_values = values * 100

    # Create pie chart with percentages to the tenth place
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct=lambda p: f'{p * sum(values) / 100:.1f}%', startangle=90,
                                      wedgeprops=dict(width=0.3, edgecolor='w'))

    # Correct the autopct to display correct percentages
    for i, a in enumerate(autotexts):
        a.set_text(f'{values[i] * 100:.1f}%')

    # Draw white circle in the center to create donut effect
    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(center_circle)

    # Add total number in the center as a percentage
    center_text = f"{total * 100:.2f}%" if is_percentage else str(total)
    plt.text(0, 0, center_text, ha='center', va='center', fontsize=20, color='black')

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    plt.title(title)
    return fig


# Streamlit title
st.title("Q2 Individual Staff CCI Data")

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)

    # Call the clean_data function
    cleaned_data = clean_data(data)

    # filtered_product_types = ["Men's Shoes", "Women's Shoes", "Unisex Shoes", "Footbeds", "Socks"]
    filtered_product_types = ["Men's Shoes", "Women's Shoes", "Footbeds", "Socks"]
    cci_sales = cleaned_data[cleaned_data['product_type'].isin(filtered_product_types)]

    # Convert the day column to datetime format
    cci_sales['day'] = pd.to_datetime(cci_sales['day'])
    # Add a 'week' column to the DataFrame
    cci_sales['week'] = cci_sales['day'].dt.to_period('W').apply(lambda r: r.start_time)

    # Filter data by product type
    # shoes_df = filter_data_by_product_type(cci_sales, ["Men's Shoes", "Women's Shoes", "Unisex Shoes"])
    shoes_df = filter_data_by_product_type(cci_sales, ["Men's Shoes", "Women's Shoes"])
    socks_df = filter_data_by_product_type(cci_sales, ["Socks"])
    insoles_df = filter_data_by_product_type(cci_sales, ["Footbeds"])

    # Aggregate data by week, location, and staff
    group_columns = ['week', 'pos_location_name', 'name_of_staff_who_helped_with_sale']
    shoes_agg = aggregate_data_by_columns(shoes_df, group_columns, 'net_quantity')
    socks_agg = aggregate_data_by_columns(socks_df, group_columns, 'net_quantity')
    insoles_agg = aggregate_data_by_columns(insoles_df, group_columns, 'net_quantity')

    # Merge aggregated data
    merged_agg = merge_aggregated_data(shoes_agg, socks_agg, insoles_agg)

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

    # '''Quarterly total CCI by location'''
    # Group by pos_location_name and sum net_quantity columns
    quarter_total = store_data_clean_df.groupby('pos_location_name')[
        ['net_quantity_shoes', 'net_quantity_socks', 'net_quantity_insoles']
    ].sum().reset_index()

    # Calculate the sock-to-shoe and insole-to-shoe ratios
    quarter_total['sock_to_shoe_ratio'] = quarter_total['net_quantity_socks'] / quarter_total['net_quantity_shoes']
    quarter_total['insole_to_shoe_ratio'] = quarter_total['net_quantity_insoles'] / quarter_total['net_quantity_shoes']

    # '''Quarterly total CCI for entire Pacers Retail'''
    # Sum net_quantity columns for the entire DataFrame
    total_quantities = store_data_clean_df[['net_quantity_shoes', 'net_quantity_socks', 'net_quantity_insoles']].sum()

    # Calculate the sock-to-shoe ratio and insole-to-shoe ratio for the total
    total_quantities['sock_to_shoe_ratio'] = total_quantities['net_quantity_socks'] / total_quantities[
        'net_quantity_shoes']
    total_quantities['insole_to_shoe_ratio'] = total_quantities['net_quantity_insoles'] / total_quantities[
        'net_quantity_shoes']

    # Convert the series to a DataFrame for better display
    total_quantities_df = total_quantities.to_frame().T

    # Total quantities
    total_shoes = quarter_total['net_quantity_shoes'].sum()
    total_socks = quarter_total['net_quantity_socks'].sum()
    total_insoles = quarter_total['net_quantity_insoles'].sum()
    total_sock_ratio = quarter_total['sock_to_shoe_ratio'].mean()
    total_insole_ratio = quarter_total['insole_to_shoe_ratio']

    col1, col2, col3 = st.columns(3)
    with col1:
        # st.write("##### Total Quantities for Quarter")
        fig = create_bar_chart1(total_shoes, total_socks, total_insoles)
        st.pyplot(fig)

    with col2:
        # st.subheader("Net Socks Ratio for Quarter")
        # sock_ratio_data = pd.concat([total_quantities_df[['sock_to_shoe_ratio']],
        #                         quarter_total[['pos_location_name', 'sock_to_shoe_ratio']]], ignore_index=True)
        # sock_ratio_data.at[0, 'pos_location_name'] = 'Total'
        # st.table(sock_ratio_data.style.set_properties(**{'font-weight': 'bold'}, subset=pd.IndexSlice[0, :]))
        total_sock_ratio = quarter_total['sock_to_shoe_ratio'].mean()
        fig = create_donut_chart(f"{total_sock_ratio:.2%}", quarter_total['sock_to_shoe_ratio'],
                                 quarter_total['pos_location_name'], 'Sock to Shoe Ratio for Quarter')
        st.pyplot(fig)

    with col3:
        # st.subheader("Net Insole Ratio for Quarter")
        # insole_ratio_data = pd.concat([total_quantities_df[['insole_to_shoe_ratio']],
        #                         quarter_total[['pos_location_name', 'insole_to_shoe_ratio']]], ignore_index=True)
        # insole_ratio_data.at[0, 'pos_location_name'] = 'Total'
        # st.table(insole_ratio_data.style.set_properties(**{'font-weight': 'bold'}, subset=pd.IndexSlice[0, :]))

        total_insole_ratio = quarter_total['insole_to_shoe_ratio'].mean()
        fig = create_donut_chart(f"{total_insole_ratio:.2%}", quarter_total['insole_to_shoe_ratio'],
                                 quarter_total['pos_location_name'], 'Insole to Shoe Ratio for Quarter')
        st.pyplot(fig)




    # Calculate the sock-to-shoe ratio
    quarter_total['sock_to_shoe_ratio'] = quarter_total['net_quantity_socks'] / quarter_total['net_quantity_shoes']
    quarter_total['insole_to_shoe_ratio'] = quarter_total['net_quantity_insoles'] / quarter_total['net_quantity_shoes']

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
        # st.subheader(f'Reports for {store_name}')
        store_df = store_dataframes[store_name]
        store_df = prepare_data(store_df)  # Ensure data is prepared after selection
        staff_members = store_df['name_of_staff_who_helped_with_sale'].unique()

        # Selectbox for staff member selection
        staff_name = st.selectbox("Select a staff member:", staff_members)

        if staff_name:
            # st.write(f'##### Report for {staff_name}')
            staff_data = store_df[store_df['name_of_staff_who_helped_with_sale'] == staff_name]

            col1, col2 = st.columns(2)

            with col1:
                create_table(staff_data)
            with col2:
                create_bar_chart(staff_data)
else:
    st.info("Please upload a CSV file to proceed.")
