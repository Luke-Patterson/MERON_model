import pandas as pd
import os

# Read all Excel files
data_dir = '../../data/linkage_data'  # Updated path to go up two directories
excel_files = ['Turkana.xlsx', 'Marsabit.xlsx', 'Isiolo.xlsx', 'Tana River.xlsx']

for file in excel_files:
    print(f"\nAnalyzing {file}:")
    df = pd.read_excel(os.path.join(data_dir, file))
    print("Columns:", list(df.columns))
    print("\nFirst few rows:")
    print(df.head())
    print("\nShape:", df.shape) 