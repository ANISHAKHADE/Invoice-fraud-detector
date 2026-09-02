import pandas as pd
import os

def build_master_dataset():
    print("1. Loading base table (invoices.csv)...")
    try:
        df = pd.read_csv("invoices.csv")
    except FileNotFoundError:
        print("Error: invoices.csv not found in the current directory.")
        return

    print("2. Joining Labels (Target Variables)...")
    try:
        labels = pd.read_csv("labels.csv")
        df = df.merge(labels, on="invoice_id", how="left")
    except FileNotFoundError:
        print("Warning: labels.csv not found. Skipping.")

    print("3. Joining Behavioral Features...")
    try:
        behaviors = pd.read_csv("behavioural_features.csv")
        df = df.merge(behaviors, on="invoice_id", how="left")
    except FileNotFoundError:
         print("Warning: behavioural_features.csv not found. Skipping.")

    print("4. Joining Image Metadata...")
    try:
        images = pd.read_csv("images_metadata.csv")
        # Drop duplicate column if it exists in both
        if 'image_path' in images.columns and 'image_path' in df.columns:
            images = images.drop(columns=['image_path'])
        df = df.merge(images, on="invoice_id", how="left")
        # Fill missing image data with 0 (since most invoices don't have images)
        if 'image_tamper_flag' in df.columns:
             df['image_tamper_flag'] = df['image_tamper_flag'].fillna(0)
        if 'ocr_total_extracted' in df.columns:
             df['ocr_total_extracted'] = df['ocr_total_extracted'].fillna(0)
    except FileNotFoundError:
         print("Warning: images_metadata.csv not found. Skipping.")

    print("5. Joining Supplier Risk Data...")
    try:
        suppliers = pd.read_csv("suppliers.csv")
        df = df.merge(suppliers, on="supplier_id", how="left")
    except FileNotFoundError:
         print("Warning: suppliers.csv not found. Skipping.")

    print("6. Joining Department Data...")
    try:
        departments = pd.read_csv("departments.csv")
        df = df.merge(departments, on="department_id", how="left")
    except FileNotFoundError:
         print("Warning: departments.csv not found. Skipping.")

    print("7. Joining Train/Test Splits...")
    try:
        splits = pd.read_csv("splits.csv")
        df = df.merge(splits, on="invoice_id", how="left")
    except FileNotFoundError:
         print("Warning: splits.csv not found. Skipping.")

    print(f"\nFinal Master Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Save the output as CSV and Parquet (Parquet is much faster for loading later)
    output_csv = "master_training_data.csv"
    output_parquet = "master_training_data.parquet"
    
    print(f"8. Saving to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    try:
        print(f"9. Saving to {output_parquet} for faster future loading...")
        df.to_parquet(output_parquet, index=False)
    except ImportError:
        print("Note: 'pyarrow' or 'fastparquet' not installed. Skipping parquet save.")
        print("You can install it with: pip install pyarrow")

    print("\nMerge complete! Data is ready for ML modeling.")

if __name__ == "__main__":
    build_master_dataset()