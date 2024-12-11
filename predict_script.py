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
        date = record['date']

        # Aggregate total predicted units for each product
        aggregated_data[product_id]['productName'] = product_name
        aggregated_data[product_id]['total_predicted_unit'] += unit_sold

        # Update store with highest unit sold prediction
        if unit_sold > aggregated_data[product_id]['highest_unit_sold']:
            aggregated_data[product_id]['store_with_highest_unit_sold_prediction'] = store
            aggregated_data[product_id]['highest_unit_sold'] = unit_sold

        # Add month (extracted from the date field)
        aggregated_data[product_id]['month'] = datetime.strptime(date, '%Y-%m-%d').strftime('%B')

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

def predict_sales(csv_file, model_path, output_file="predictions_stock.json"):
    """Predict sales for the next month and process the data after predictions."""
    # Load the trained model
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    # yyyy-mm-dd
    # Parse the date column with dayfirst=True to handle DD-MM-YYYY format
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
# p001
# for loop i=0 i+1
    # Convert 'item' column to integer using a mapping
    item_mapping = {item: idx + 1 for idx, item in enumerate(df['item'].unique())}
    df['item'] = df['item'].map(item_mapping)

    # Dynamically determine the last date in the dataset
    last_date = df['date'].max() # 31-12-2018

    # Calculate the first date of the next month
    next_month_start = (last_date + MonthBegin(1))

    # Generate all dates for the next month
    next_month_days = pd.date_range(next_month_start, periods=31, freq='D')

    # Dynamically extract unique stores and items
    stores = df['store'].unique()  # Get unique store IDs
    items = df['item'].unique()  # Get unique item IDs

    # Create all combinations of future dates, stores, and items
    future_df = pd.DataFrame(
        [(date, store, item) for date in next_month_days for store in stores for item in items],
        columns=['date', 'store', 'item']
    )

    # Merge product details (like product_name) dynamically if available
    if 'product_name' in df.columns:
        future_df = future_df.merge(df[['item', 'product_name']].drop_duplicates(), on='item', how='left')

    # Concatenate the original and future data for preprocessing
    combined_df = pd.concat([df, future_df], ignore_index=True)
    # normalize to 0-1 100 -- 0.9
    # Add lag features
    combined_df = lag_features(combined_df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

    # Add rolling mean features
    combined_df = roll_mean_features(combined_df, [365, 546, 730])

    # Add exponential weighted mean features
    combined_df = ewm_features(combined_df, [0.99, 0.95, 0.9, 0.8, 0.7, 0.5], [91, 98, 105, 112, 180, 270, 365, 546, 728])

    # Extract day_of_week and month features
    combined_df['day_of_week'] = combined_df['date'].dt.dayofweek
    combined_df['month'] = combined_df['date'].dt.month
    combined_df = pd.get_dummies(combined_df, columns=['day_of_week', 'month'])

    # Align combined_df with the model's expected features
    train_columns = model.feature_name()  # Extract feature names from the model
    missing_cols = set(train_columns) - set(combined_df.columns)  # Find missing columns
    for col in missing_cols:
        combined_df[col] = 0  # Add missing columns with default value 0

    combined_df = combined_df[train_columns]  # Reorder columns to match the model

    # Make predictions for the future dataset
    future_predictions = combined_df[len(df):]  # Slice to get only the future rows
    future_df['unitSold'] = np.expm1(model.predict(future_predictions))  # Reverse log transformation

    # Prepare the final result with all required fields
    future_df['id'] = range(1, len(future_df) + 1)  # Add an 'id' column for unique identification
    result = future_df[['id', 'date', 'store', 'item', 'product_name', 'unitSold']]  # Prepare the result

    # Convert the 'date' column to string format (YYYY-MM-DD)
    result['date'] = result['date'].dt.strftime('%Y-%m-%d')

    # Save the result as a JSON file
    result.to_json(output_file, orient="records", indent=4)
    print(f"Predictions saved to {output_file}")

    # Call process_sales_data to aggregate the predictions
    final_result = process_sales_data(output_file, "final_predictions.json")
    print()
    # Now apply constraints on the predictions
    def constrain_prediction(row, df):
        # Sum up all 'unitSold' values for the same 'item', ignoring 'store'
        original_unit_sold = df.loc[(df['item'] == row['productId']), 'unitSold'].sum()
        print(original_unit_sold)
        if pd.notna(original_unit_sold):  # Ensure original value exists
            # Calculate the lower and upper bounds based on ±6% of the summed unitSold
            lower_bound = max(0, original_unit_sold * 0.90)  # 6% lower limit
            upper_bound = original_unit_sold * 1.02  # 6% upper limit
            print(lower_bound,upper_bound)
            # Clamp the prediction value within the bounds
            return max(lower_bound, min(row['total_predicted_unit'], upper_bound))

        return row['total_predicted_unit']  # Leave as is if no original unitSold found
# 2785 1650
    # Apply constraints to each prediction in the final_result
    final_result = [dict(row, unitSold=constrain_prediction(row, df)) for row in final_result]

    # Save the final result after applying constraints
    final_output_file = "final_predictions_with_constraints.json"
    with open(final_output_file, 'w') as f:
        json.dump(final_result, f, indent=4)

    print(f"Final predictions with constraints saved to {final_output_file}")

    # Return the final processed predictions
    return final_result
