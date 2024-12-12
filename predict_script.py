import pandas as pd
import numpy as np
import pickle
import json
from pandas.tseries.offsets import MonthBegin
from Preprocessing_script import lag_features, roll_mean_features, random_noise, ewm_features
from collections import defaultdict
from datetime import datetime
#
#
def process_sales_data(input_file, output_file):
    """
    Process sales data from a JSON file and generate output with aggregated predictions.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
    """
    # Read the input JSON file
    with open(input_file, 'r') as f:
        sales_data = json.load(f)

    # Dictionary to store aggregated data for each product
    aggregated_data = defaultdict(lambda: {
        'productName': None,
        'total_predicted_unit': 0,
        'store_with_highest_unit_sold_prediction': None,
        'highest_unit_sold': 0
    })

    # Process each record in the sales data
    for record in sales_data:
        product_id = record['item']
        store = record['store']
        product_name = record['product_name']
        unit_sold = record['unitSold']
        month = record['month']
        location = record['location']

        # Aggregate total predicted units for each product
        aggregated_data[product_id]['productName'] = product_name
        aggregated_data[product_id]['total_predicted_unit'] += unit_sold

        # Update store with highest unit sold prediction
        if unit_sold > aggregated_data[product_id]['highest_unit_sold']:
            aggregated_data[product_id]['store_with_highest_unit_sold_prediction'] = store
            aggregated_data[product_id]['highest_unit_sold'] = unit_sold

        # Add month (extracted from the date field)
        aggregated_data[product_id]['month'] = month
        aggregated_data[product_id]['location'] = location

    # Prepare the final output format
    result = []
    for product_id, data in aggregated_data.items():
        result.append({
            'productId': product_id,
            'productName': data['productName'],
            'total_predicted_unit': data['total_predicted_unit'],
            'month': data['month'],
            'store_with_highest_unit_sold_prediction': data['store_with_highest_unit_sold_prediction']
        })

    # Write the result to the output JSON file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Processed data saved to {output_file}")
    return result  # Return the final result



import pandas as pd
import pickle
import numpy as np
from pandas.tseries.offsets import MonthBegin
from datetime import datetime

# def predict_sales(csv_file, model_path, output_file="predictions_stock.json"):
#     """
#     Predict sales for the next month, aggregate predictions by store and product, apply constraints,
#     and process the data.
#     """
#     # Load the trained model
#     with open(model_path, "rb") as file:
#         model = pickle.load(file)
#
#     # Read the uploaded CSV file into a DataFrame
#     df = pd.read_csv(csv_file)
#
#     # Clean the date column (strip spaces and remove non-ASCII characters)
#     df['date'] = df['date'].str.strip()  # Removes leading/trailing spaces
#     df['date'] = df['date'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))  # Removes non-ASCII characters
#
#     # Parse the date column (YYYY-MM-DD format)
#     df['date'] = pd.to_datetime(df['date'], errors='coerce')
#
#     # Identify invalid dates (NaT)
#     invalid_dates = df[df['date'].isna()]
#     if not invalid_dates.empty:
#         print("Invalid dates found in the input file:")
#         print(invalid_dates)
#         # Handle invalid dates - one approach is to fill them with the previous valid date or current date
#         # Option 1: Replace NaT with the previous valid date (if it exists)
#         df['date'] = df['date'].fillna(method='ffill')  # Forward fill
#         print("Invalid dates corrected.")
#
#     # Map product IDs to integers dynamically
#     item_mapping = {item: idx + 1 for idx, item in enumerate(df['item'].unique())}
#     df['item'] = df['item'].map(item_mapping)
#
#     # Create a store-location mapping (ensure each store has a unique location)
#     store_location_mapping = df[['store', 'location']].drop_duplicates().set_index('store')['location'].to_dict()
#
#     # Dynamically determine the last date in the dataset
#     last_date = df['date'].max()
#
#     # Generate the future dates for prediction
#     next_month_start = (last_date + MonthBegin(1))
#     next_month_days = pd.date_range(next_month_start, periods=31, freq='D')
#
#     # Dynamically extract unique stores and items
#     stores = df['store'].unique()
#     items = df['item'].unique()
#
#     # Create all combinations of future dates, stores, and items
#     future_df = pd.DataFrame(
#         [(date, store, item) for date in next_month_days for store in stores for item in items],
#         columns=['date', 'store', 'item']
#     )
#
#     # Merge product details dynamically
#     if 'product_name' in df.columns:
#         future_df = future_df.merge(df[['item', 'product_name']].drop_duplicates(), on='item', how='left')
#
#     # Map the 'location' for each store in future_df
#     future_df['location'] = future_df['store'].map(store_location_mapping)
#
#     # If a store's location is not found, fill with 'Unknown' or a default value
#     future_df['location'].fillna('Unknown', inplace=True)
#
#     # Preprocess and add features (lag, rolling mean, etc.)
#     combined_df = pd.concat([df, future_df], ignore_index=True)
#     combined_df = lag_features(combined_df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
#     combined_df = roll_mean_features(combined_df, [365, 546, 730])
#     combined_df = ewm_features(combined_df, [0.99, 0.95, 0.9, 0.8, 0.7, 0.5], [91, 98, 105, 112, 180, 270, 365, 546, 728])
#     combined_df['day_of_week'] = combined_df['date'].dt.dayofweek
#     combined_df['month'] = combined_df['date'].dt.month
#     combined_df = pd.get_dummies(combined_df, columns=['day_of_week', 'month'])
#
#     # Align columns with model features
#     train_columns = model.feature_name()
#     missing_cols = set(train_columns) - set(combined_df.columns)
#     for col in missing_cols:
#         combined_df[col] = 0
#     combined_df = combined_df[train_columns]
#
#     # Make predictions
#     future_predictions = combined_df[len(df):]
#     future_df['unitSold'] = np.expm1(model.predict(future_predictions))
#
#     # Aggregate predictions by store, product, and month, including product_name
#     future_df['month'] = future_df['date'].dt.strftime('%B')
#     future_df.to_json("inital_csv", orient="records", indent=4)
#
#     # Aggregate predictions by store, product, and month
#     aggregated_df = future_df.groupby(['store', 'item', 'product_name', 'month', 'location'], as_index=False).agg(
#         total_unitSold=('unitSold', 'sum')
#     )
#     aggregated_df.to_json("inital_csv.json", orient="records", indent=4)
#     cnt=0
#     # Apply constraints to aggregated predictions
#     def apply_constraints(row, cnt=0):
#         original_unit_sold = df[df['item'] == row['item']]['unitSold'].sum()
#         cnt=cnt+1
#         print(original_unit_sold,cnt)
#         if pd.notna(original_unit_sold):
#             lower_bound = max(0, original_unit_sold * 0.90)  # 90% lower limit
#             upper_bound = original_unit_sold * 1.02  # 102% upper limit
#             return max(lower_bound, min(row['total_unitSold'], upper_bound))
#         return row['total_unitSold']
#
#     aggregated_df['constrained_unitSold'] = aggregated_df.apply(apply_constraints, axis=1)
#
#     # Prepare data for process_sales_data
#     final_df = aggregated_df[['store', 'item', 'product_name', 'month', 'constrained_unitSold', 'location']].rename(
#         columns={'constrained_unitSold': 'unitSold'}
#     )
#
#     # Save the aggregated and constrained predictions
#     final_output_file = "aggregated_predictions.json"
#     final_df.to_json(final_output_file, orient="records", indent=4)
#     print(f"Aggregated predictions saved to {final_output_file}")
#
#     # Process sales data
#     final_result = process_sales_data(final_output_file, "final_predictions.json")
#     print("Final predictions processed successfully.")
#     return final_result

def predict_sales(csv_file, model_path, output_file="predictions_stock.json"):
    """
    Predict sales for the next month, aggregate predictions by store and product, apply constraints,
    and process the data.
    """
    # Load the trained model
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Clean the date column (strip spaces and remove non-ASCII characters)
    df['date'] = df['date'].str.strip()  # Removes leading/trailing spaces
    df['date'] = df['date'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))  # Removes non-ASCII characters

    # Parse the date column (YYYY-MM-DD format)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Identify invalid dates (NaT)
    invalid_dates = df[df['date'].isna()]
    if not invalid_dates.empty:
        print("Invalid dates found in the input file:")
        print(invalid_dates)
        # Handle invalid dates - one approach is to fill them with the previous valid date or current date
        # Option 1: Replace NaT with the previous valid date (if it exists)
        df['date'] = df['date'].fillna(method='ffill')  # Forward fill
        print("Invalid dates corrected.")

    # Map product IDs to integers dynamically
    item_mapping = {item: idx + 1 for idx, item in enumerate(df['item'].unique())}
    df['item'] = df['item'].map(item_mapping)

    # Create a store-location mapping (ensure each store has a unique location)
    store_location_mapping = df[['store', 'location']].drop_duplicates().set_index('store')['location'].to_dict()

    # Dynamically determine the last date in the dataset
    last_date = df['date'].max()

    # Generate the future dates for prediction
    next_month_start = (last_date + MonthBegin(1))
    next_month_days = pd.date_range(next_month_start, periods=31, freq='D')

    # Dynamically extract unique stores and items
    stores = df['store'].unique()
    items = df['item'].unique()

    # Create all combinations of future dates, stores, and items
    future_df = pd.DataFrame(
        [(date, store, item) for date in next_month_days for store in stores for item in items],
        columns=['date', 'store', 'item']
    )

    # Merge product details dynamically
    if 'product_name' in df.columns:
        future_df = future_df.merge(df[['item', 'product_name']].drop_duplicates(), on='item', how='left')

    # Map the 'location' for each store in future_df
    future_df['location'] = future_df['store'].map(store_location_mapping)

    # If a store's location is not found, fill with 'Unknown' or a default value
    future_df['location'].fillna('Unknown', inplace=True)

    # Preprocess and add features (lag, rolling mean, etc.)
    combined_df = pd.concat([df, future_df], ignore_index=True)
    combined_df = lag_features(combined_df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
    combined_df = roll_mean_features(combined_df, [365, 546, 730])
    combined_df = ewm_features(combined_df, [0.99, 0.95, 0.9, 0.8, 0.7, 0.5], [91, 98, 105, 112, 180, 270, 365, 546, 728])
    combined_df['day_of_week'] = combined_df['date'].dt.dayofweek
    combined_df['month'] = combined_df['date'].dt.month
    combined_df = pd.get_dummies(combined_df, columns=['day_of_week', 'month'])

    # Align columns with model features
    train_columns = model.feature_name()
    missing_cols = set(train_columns) - set(combined_df.columns)
    for col in missing_cols:
        combined_df[col] = 0
    combined_df = combined_df[train_columns]

    # Make predictions
    future_predictions = combined_df[len(df):]
    future_df['unitSold'] = np.expm1(model.predict(future_predictions))

    # Aggregate predictions by store, product, and month, including product_name
    future_df['month'] = future_df['date'].dt.strftime('%B')
    future_df.to_json("inital_csv", orient="records", indent=4)

    # Aggregate predictions by store, product, and month
    aggregated_df = future_df.groupby(['store', 'item', 'product_name', 'month', 'location'], as_index=False).agg(
        total_unitSold=('unitSold', 'sum')
    )
    aggregated_df.to_json("inital_csv.json", orient="records", indent=4)

    # Prepare data for process_sales_data
    final_df = aggregated_df[['store', 'item', 'product_name', 'month', 'total_unitSold', 'location']].rename(
        columns={'total_unitSold': 'unitSold'}
    )

    # Save the aggregated predictions
    final_output_file = "aggregated_predictions.json"
    final_df.to_json(final_output_file, orient="records", indent=4)
    print(f"Aggregated predictions saved to {final_output_file}")

    # Process sales data
    final_result = process_sales_data(final_output_file, "final_predictions.json")
    print("Final predictions processed successfully.")

    # Return both final_prediction and aggregated_prediction as a JSON object
    final_prediction = final_result  # Assuming the final_result is the desired final prediction
    aggregated_prediction = aggregated_df.to_dict(orient='records')

    return {
        'product': final_prediction,
        'store_aggregated_prediction': aggregated_prediction
    }
