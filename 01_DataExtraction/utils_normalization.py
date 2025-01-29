import os
import glob
import pandas as pd


source_folder_path = '/work/AML_Project/Data/data_extractedframes_entire' 
destination_folder_path = '/work/AML_Project/Data/normalized_csvs' 
os.makedirs(destination_folder_path, exist_ok=True)
csv_files = glob.glob(os.path.join(source_folder_path, '*.csv'))

for file_path in csv_files:
    df = pd.read_csv(file_path)
    
    required_cols = {'Frames', 'bvp', 'eda'}
    if not required_cols.issubset(df.columns):
        print(f"Skipping {file_path}: missing required columns.")
        continue
    
    print(f"Normalizing: {file_path}")
    # Normalize 'bvp' and 'eda' columns using Min-Max normalization
        # normalize each csv separately
    for col in ['bvp', 'eda']:
        min_val = df[col].min() # Calculates the minimum value in the current column 
        max_val = df[col].max() # Calculates the maximum value in the current column 
        if max_val - min_val != 0:
            df[col] = (df[col] - min_val) / (max_val - min_val) # scaling the column values to a range of 0 to 1
        else:
            df[col] = 0.0  

    filename = os.path.basename(file_path)
    new_file_path = os.path.join(destination_folder_path, filename)

    df.to_csv(new_file_path, index=False)

    print(f"Processed and normalized: {new_file_path}")
