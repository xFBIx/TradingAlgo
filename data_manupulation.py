import pandas as pd


# Function to process the CSV file
def clean_csv(csv_file_path, output_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Remove trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Remove commas from all columns and convert numeric columns to appropriate data types
    df = df.replace({",": ""}, regex=True)

    # Rename columns to have the first letter capital and others small
    df.columns = [col.capitalize() for col in df.columns]

    # Save the cleaned DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False)
    print(f"File saved to {output_file_path}")


# Example usage
# csv_file_path = "GPPL.csv"  # Input file path
# output_file_path = "GPPL_cleaned.csv"  # Output file path

# clean_csv(csv_file_path, output_file_path)
