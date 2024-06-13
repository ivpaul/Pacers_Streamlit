import pandas as pd
import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="Inventory",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize 'Required Stock' with different minimums
def get_required_stock(row):
    if pd.Series(row['Title']).str.contains('Currex RunPro', case=False, na=False).any():
        return max(5, row['Pacers Georgetown'])
    elif pd.Series(row['Title']).str.contains('Currex SupportSTP', case=False, na=False).any():
        return max(3, row['Pacers Georgetown'])
    else:
        return row['Pacers Georgetown']

# Streamlit title
st.title("Inventory")

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    all_inventory_df= pd.read_csv(uploaded_file, low_memory=False)
    pd.set_option('display.max_columns', None)
    columns_of_interest = [
        'Title',
        'Option1 Value',
        'Option2 Value',
        'SKU',
        'Online Store WH (Saltbox)',
        'Pacers 14th St',
        'Pacers Alexandria',
        'Pacers Arlington',
        'Pacers Georgetown',
        'Pacers Navy Yard'
    ]

    # Make new dataframe
    current_inventory_df = all_inventory_df[columns_of_interest]

    # Rename columns
    current_inventory_df = current_inventory_df.rename(columns={
        'Option1 Value': 'Size',
        'Option2 Value': 'Color'
    })

    # Drop null values
    current_inventory_df = current_inventory_df.dropna()

    # Drop duplicates
    current_inventory_df = current_inventory_df.drop_duplicates()
    duplicates = current_inventory_df.duplicated().sum()

    # Replace 'not stocked' with 0
    current_inventory_df = current_inventory_df.replace('not stocked', 0)

    # Convert relevant columns to integers
    columns_to_convert = [
        'Online Store WH (Saltbox)',
        'Pacers 14th St',
        'Pacers Alexandria',
        'Pacers Arlington',
        'Pacers Georgetown',
        'Pacers Navy Yard'
    ]

    for column in columns_to_convert:
        current_inventory_df[column] = pd.to_numeric(current_inventory_df[column], errors='coerce').fillna(0).astype(
            int)

    filtered_df = current_inventory_df[
        current_inventory_df['Title'].str.contains('Currex|Feetures', case=False, na=False)
    ]

    filtered_df = filtered_df.sort_values(by='SKU', ascending=True)

    # Filter rows where 'Online Store WH (Saltbox)' is greater than 0
    SB_filtered_df = filtered_df[filtered_df['Online Store WH (Saltbox)'] > 0]

    # Create currex_df by filtering SB_filtered_df for 'Currex' in 'Title'
    currex_df = SB_filtered_df[SB_filtered_df['Title'].str.contains('Currex SupportSTP|Currex RunPro', case=False, na=False)]

    # Reset the index of currex_df
    currex_df = currex_df.reset_index(drop=True)

    socks_df = SB_filtered_df[
        SB_filtered_df['Title'].str.contains('Feetures', case=False, na=False) &
        SB_filtered_df['Color'].str.lower().isin(['white', 'black', 'grey'])
        ]

    # Reset the index of sock_df
    socks_df = socks_df.reset_index(drop=True)

    # Display dataframes
    st.subheader("Currex Inventory")
    st.dataframe(currex_df)

    # st.subheader("Socks Inventory")
    # st.dataframe(socks_df)

    # Input restock requirements using a data editor
    st.subheader("Restock Requirements")

    # Create a table for restock input
    restock_data = currex_df[['SKU', 'Title', 'Size', 'Color', 'Online Store WH (Saltbox)', 'Pacers Georgetown']].copy()

    # Initialize 'Required Stock'
    restock_data['Required Stock'] = restock_data.apply(get_required_stock, axis=1)

    # Display the data editor for restock input
    edited_restock_data = st.data_editor(restock_data, use_container_width=True)

    # Calculate restock amounts
    restock_requests = []
    for index, row in edited_restock_data.iterrows():
        georgetown_stock = row['Pacers Georgetown']
        required_stock = row['Required Stock']
        saltbox_stock = row['Online Store WH (Saltbox)']

        # Calculate the maximum possible restock amount based on available stock
        restock_amount = min(max(0, required_stock - georgetown_stock), saltbox_stock)

        if restock_amount > 0:
            restock_requests.append({
                'SKU': row['SKU'],
                'Title': row['Title'],
                'Size': row['Size'],
                'Color': row['Color'],
                # 'Current Stock at GT': row['Pacers Georgetown'],
                'Requested Amount': restock_amount
            })

    # Create restock DataFrame
    if restock_requests:
        restock_df = pd.DataFrame(restock_requests)
        st.subheader("Restock Requests for GT")
        st.dataframe(restock_df)

        # Export to CSV
        csv = restock_df.to_csv(index=False)
        st.download_button("Download Restock Requests CSV", csv, "restock_requests.csv", "text/csv")

else:
    st.info("Please upload a CSV file to proceed.")

