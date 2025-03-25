import os
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MalnutritionFlagCreator:
    def __init__(self):
        # Paths
        self.data_dir = 'data'
        self.features_path = os.path.join(self.data_dir, 'processed/resnet50_features.csv')
        self.metadata_dir = os.path.join(self.data_dir, 'linkage_data')
        self.output_dir = os.path.join(self.data_dir, 'processed')
        
        # WHO Standards thresholds
        self.whz_severe = -3.0  # Severe acute malnutrition WHZ threshold
        self.whz_moderate = -2.0  # Moderate acute malnutrition WHZ threshold
        self.muac_severe = 11.5  # Severe acute malnutrition MUAC threshold (cm)
        self.muac_moderate = 12.5  # Moderate acute malnutrition MUAC threshold (cm)
        
        # Load WHO z-score tables
        self.who_tables = {}
        self.load_who_tables()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_who_tables(self):
        """Load WHO z-score reference tables"""
        logging.info("Loading WHO z-score tables...")
        
        # File mapping
        files = {
            'wfh_boys': 'wfh_boys.csv',
            'wfh_girls': 'wfh_girls.csv',
            'wfl_boys': 'wfl_boys.csv',
            'wfl_girls': 'wfl_girls.csv'
        }
        
        for key, filename in files.items():
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                self.who_tables[key] = pd.read_csv(file_path)
                logging.info(f"Loaded {filename}")
            else:
                logging.warning(f"WHO table not found: {filename}")
    
    def _find_nearest_height(self, table, height):
        """Find the nearest height/length value in WHO table"""
        height_col = 'Height' if 'Height' in table.columns else 'Length'
        idx = (np.abs(table[height_col].values - float(height))).argmin()
        return idx
    
    def _calc_zscore(self, measure, idx, weight):
        """Calculate z-score using WHO LMS method"""
        try:
            l, m, s = self.who_tables[measure].iloc[idx][['L', 'M', 'S']].values
            
            if l == 0:
                zscore = np.log(weight / m) / s
            else:
                zscore = (np.power(weight / m, l) - 1) / (l * s)
            
            return np.round(zscore, 2)
        except:
            return np.nan
    
    def calculate_wfh_zscore(self, row):
        """
        Calculate Weight-for-Height Z-score using WHO standards
        Uses the appropriate table based on gender and age
        """
        try:
            weight = row['weight_kg']
            height = row['height_cm']
            age_months = row['age_months']
            gender = row['gender'].lower() if isinstance(row['gender'], str) else 'male'
            
            # Determine which table to use based on gender
            if gender in ['m', 'male', '1']:
                table_prefix = 'wfh_boys' if age_months > 24 else 'wfl_boys'
            else:
                table_prefix = 'wfh_girls' if age_months > 24 else 'wfl_girls'
            
            # Find nearest height/length in table
            idx = self._find_nearest_height(self.who_tables[table_prefix], height)
            
            # Calculate z-score
            return self._calc_zscore(table_prefix, idx, weight)
        except Exception as e:
            logging.warning(f"Error calculating WHZ: {str(e)}")
            return np.nan
    
    def load_data(self):
        """Load features and metadata"""
        logging.info("Loading data...")
        
        # Load features
        features_df = pd.read_csv(self.features_path)
        logging.info(f"Loaded features with shape: {features_df.shape}")
        
        # Load and combine metadata from all counties
        metadata_dfs = []
        excel_files = ['Turkana.xlsx', 'Marsabit.xlsx', 'Isiolo.xlsx', 'Tana River.xlsx']
        
        for file in excel_files:
            file_path = os.path.join(self.metadata_dir, file)
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                metadata_dfs.append(df)
                logging.info(f"Loaded {file} with {len(df)} records")
        
        metadata_df = pd.concat(metadata_dfs, ignore_index=True)
        logging.info(f"Combined metadata shape: {metadata_df.shape}")
        
        return features_df, metadata_df
    
    def create_malnutrition_flags(self, metadata_df):
        """Calculate malnutrition flags based on anthropometric measurements"""
        logging.info("Calculating malnutrition flags")
        
        # Create copy to avoid modifying original
        df = metadata_df.copy()
        
        # Calculate WHZ for each child
        df['whz'] = df.apply(lambda row: self.calculate_wfh_zscore(row), axis=1)
        
        # Calculate flags
        df['sam_whz'] = df['whz'] < self.whz_severe
        df['mam_whz'] = (df['whz'] >= self.whz_severe) & (df['whz'] < self.whz_moderate)
        
        df['sam_muac'] = df['muac_cm'] < self.muac_severe
        df['mam_muac'] = (df['muac_cm'] >= self.muac_severe) & (df['muac_cm'] < self.muac_moderate)
        
        # Combined flags (any form of malnutrition)
        df['sam'] = df['sam_whz'] | df['sam_muac']
        df['mam'] = df['mam_whz'] | df['mam_muac']
        df['malnutrition'] = df['sam'] | df['mam']
        
        # Convert boolean flags to int for easier processing
        bool_columns = ['sam_whz', 'mam_whz', 'sam_muac', 'mam_muac', 'sam', 'mam', 'malnutrition']
        for col in bool_columns:
            df[col] = df[col].astype(int)
            
        return df
    
    def create_final_dataset(self, features_df, metadata_df):
        """Create final dataset with features and malnutrition flags"""
        # Calculate malnutrition flags
        metadata_with_flags = self.create_malnutrition_flags(metadata_df)
        
        # Select relevant columns from metadata
        keep_columns = [
            'photo_id', 'county', 'age_months', 'gender',
            'weight_kg', 'height_cm', 'muac_cm', 'whz',
            'sam_whz', 'mam_whz', 'sam_muac', 'mam_muac',
            'sam', 'mam', 'malnutrition'
        ]
        metadata_subset = metadata_with_flags[keep_columns]
        
        # Merge with features
        final_df = pd.merge(
            features_df,
            metadata_subset,
            on='photo_id',
            how='inner'
        )
        
        # Log statistics
        logging.info("\nMalnutrition Statistics:")
        logging.info(f"Total children: {len(final_df)}")
        logging.info(f"SAM cases: {final_df['sam'].sum()} ({final_df['sam'].mean()*100:.1f}%)")
        logging.info(f"MAM cases: {final_df['mam'].sum()} ({final_df['mam'].mean()*100:.1f}%)")
        logging.info(f"Total malnourished: {final_df['malnutrition'].sum()} ({final_df['malnutrition'].mean()*100:.1f}%)")
        
        # Save datasets
        flags_only = metadata_subset.copy()
        features_with_flags = final_df.copy()
        
        flags_path = os.path.join(self.output_dir, 'malnutrition_flags.csv')
        final_path = os.path.join(self.output_dir, 'features_with_flags.csv')
        
        flags_only.to_csv(flags_path, index=False)
        features_with_flags.to_csv(final_path, index=False)
        
        logging.info(f"\nSaved files:")
        logging.info(f"1. Malnutrition flags only: {flags_path}")
        logging.info(f"2. Features with flags: {final_path}")
        
        return final_df

def main():
    try:
        processor = MalnutritionFlagCreator()
        
        # Load data
        features_df, metadata_df = processor.load_data()
        
        # Create final dataset
        final_df = processor.create_final_dataset(features_df, metadata_df)
        
        logging.info("\nProcessing completed successfully!")
        logging.info(f"Final dataset shape: {final_df.shape}")
        
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 