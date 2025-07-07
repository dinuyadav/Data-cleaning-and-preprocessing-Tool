import pandas as pd
import numpy as np

# Load dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully.")
        return df
    except Exception as e:
        print("❌ Error loading file:", e)
        return None

# Handle missing values
def handle_missing_values(df):
    print("\n🔍 Handling missing values...")
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before: {missing_before}")

    # Replace missing values with column mean (for numerical)
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column].fillna(df[column].mean(), inplace=True)

    # Replace missing values with mode (for categorical)
    for column in df.select_dtypes(include=['object']).columns:
        df[column].fillna(df[column].mode()[0], inplace=True)

    missing_after = df.isnull().sum().sum()
    print(f"Missing values after: {missing_after}")
    return df

# Remove duplicate rows
def remove_duplicates(df):
    print("\n🧹 Removing duplicate rows...")
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Duplicates removed: {before - after}")
    return df

# Detect and remove outliers using IQR
def remove_outliers(df):
    print("\n📊 Detecting and removing outliers using IQR method...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        before = df.shape[0]
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        after = df.shape[0]
        print(f"  → {col}: Removed {before - after} outliers")
    return df

# Save cleaned data
def save_cleaned_data(df, output_path):
    try:
        df.to_csv(output_path, index=False)
        print(f"\n✅ Cleaned dataset saved to: {output_path}")
    except Exception as e:
        print("❌ Error saving cleaned data:", e)

# Main program
if __name__ == "__main__":
    file_path = r"C:\Users\91880\Desktop\Data_Cleaning_Tool\sample_dataset.csv"
    output_path = r"C:\Users\91880\Desktop\Data_Cleaning_Tool\cleaned_dataset.csv"

    df = load_data(file_path)

    
    if df is not None:
        print("\n🔎 Initial data preview:\n", df.head(7))

        df = handle_missing_values(df)
        df = remove_duplicates(df)
        df = remove_outliers(df)

        print("\n✅ Final cleaned data preview:\n", df.head(7))

        save_cleaned_data(df, output_path)
