import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for data processing parameters"""
    # File paths
    input_file: str = './MIDASCDWH_DATA_2024-11-12_1309.csv'
    output_file: str = './whodas_outcomes.csv'
    
    # Score thresholds
    score_increase_threshold: int = 5
    high_score_threshold: int = 16
    
    # Time windows
    three_month_days: int = 90
    six_month_days: int = 180
    
    # Age groups and LOS thresholds
    age_groups: List[Tuple[float, float]] = None
    los_thresholds: List[int] = None
    
    def __post_init__(self):
        if self.age_groups is None:
            self.age_groups = [
                (0, 71),    # Group 1
                (71, 79),   # Group 2
                (79, 88),   # Group 3
                (88, float('inf'))  # Group 4
            ]
        if self.los_thresholds is None:
            self.los_thresholds = [5, 6, 6, 7]

class WHODASProcessor:
    """Class for processing WHODAS data and creating outcome variables"""
    
    def __init__(self, config: Config):
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.df_whodas: Optional[pd.DataFrame] = None
        self.final_df: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate input data"""
        logger.info(f"Loading data from {self.config.input_file}...")
        if not os.path.exists(self.config.input_file):
            logger.error(f"File {self.config.input_file} does not exist!")
            raise FileNotFoundError(f"File {self.config.input_file} does not exist!")
        
        return pd.read_csv(self.config.input_file)
    
    def extract_timepoint_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Extract data for each time point"""
        logger.info("Extracting data for each time point...")
        
        # Define columns for each time point
        m0_include_cols = ["record_id", "hospital_los"] + [f's{i}' for i in range(1, 13)]
        m3_include_cols = ['record_id', 'score1_90'] + [f's{i}_90' for i in range(1, 13)]
        m6_include_cols = ['record_id', 'score1_180'] + [f's{i}_180' for i in range(1, 13)]
        
        # Extract data for each time point
        index_df = self.df[self.df['redcap_event_name'] == 'index_surgery_arm_1'][m0_include_cols].copy()
        m3_df = self.df[self.df['redcap_event_name'] == '3months_arm_1'][m3_include_cols].copy()
        m6_df = self.df[self.df['redcap_event_name'] == '6months_arm_1'][m6_include_cols].copy()
        
        return index_df, m3_df, m6_df
    
    def scale_and_fill_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale scores to 0-100 and handle missing values"""
        logger.info("Scaling scores and handling missing values...")
        df_scaled = df.copy()
        
        # Handle score1 missing values
        missing_count_score1 = df_scaled['score1'].isna().sum()
        if missing_count_score1 > 0:
            logger.info(f"Filling score1 missing values: {missing_count_score1} values set to 0")
            df_scaled['score1'] = df_scaled['score1'].fillna(0)
        
        # Scale scores
        for col in ['score1', 'score1_90', 'score1_180']:
            if col in df_scaled.columns:
                df_scaled[col] = df_scaled[col] / 60 * 100
        
        # Fill missing values
        df_scaled['score1_90'] = df_scaled['score1_90'].fillna(df_scaled['score1'])
        df_scaled['score1_180'] = df_scaled['score1_180'].fillna(df_scaled['score1_90'])
        df_scaled['days_death_surgery'] = df_scaled['days_death_surgery'].fillna(0)
        
        # Handle hospital_los missing values
        if 'hospital_los' in df_scaled.columns:
            missing_count_los = df_scaled['hospital_los'].isna().sum()
            if missing_count_los > 0:
                logger.info(f"Filling hospital_los missing values: {missing_count_los} values set to 0")
                df_scaled['hospital_los'] = df_scaled['hospital_los'].fillna(0)
        
        return df_scaled
    
    def create_outcome_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create outcome variables for 3 and 6 months"""
        logger.info("Creating outcome variables...")
        df_outcome = df.copy()
        
        # Ensure score1 has no missing values
        if df_outcome['score1'].isna().any():
            logger.warning("score1 has missing values, filling with 0")
            df_outcome['score1'] = df_outcome['score1'].fillna(0)
        
        # Create 3-month outcome
        df_outcome['dd_3month'] = (
            ((df_outcome['days_death_surgery'] >= 1) & 
             (df_outcome['days_death_surgery'] <= self.config.three_month_days)) |
            ((df_outcome['score1_90'] - df_outcome['score1']) >= self.config.score_increase_threshold) &
            (df_outcome['score1_90'].notna()) |
            (df_outcome['score1_90'] >= self.config.high_score_threshold)
        ).astype(int)
        
        # Create 6-month outcome
        df_outcome['dd_6month'] = (
            ((df_outcome['days_death_surgery'] >= 1) & 
             (df_outcome['days_death_surgery'] <= self.config.six_month_days)) |
            ((df_outcome['score1_180'] - df_outcome['score1']) >= self.config.score_increase_threshold) &
            (df_outcome['score1_90'].notna()) |
            (df_outcome['score1_180'] >= self.config.high_score_threshold)
        ).astype(int)
        
        # Create los_target based on age groups
        df_outcome = self._create_los_target(df_outcome)
        
        return df_outcome
    
    def _create_los_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create length of stay target based on age groups"""
        logger.info("Creating length of stay target...")
        
        # Get age data if not present
        if 'age' not in df.columns:
            age_data = self.df[self.df["redcap_event_name"].str.contains("index")][['record_id', 'age']]
            df = pd.merge(df, age_data, on='record_id', how='left')
            df['age'] = df['age'].fillna(df['age'].mean())
        
        # Initialize los_target
        df['los_target'] = 0
        
        # Set los_target based on age groups and thresholds
        for (min_age, max_age), threshold in zip(self.config.age_groups, self.config.los_thresholds):
            mask = (df['age'] >= min_age) & (df['age'] < max_age) & (df['hospital_los'] > threshold)
            df.loc[mask, 'los_target'] = 1
            
            # Log statistics
            group_size = ((df['age'] >= min_age) & (df['age'] < max_age)).sum()
            los_target_count = mask.sum()
            if group_size > 0:
                percentage = los_target_count / group_size * 100
                logger.info(f"Age group {min_age}-{max_age} (threshold>{threshold} days): "
                          f"{los_target_count}/{group_size} ({percentage:.2f}%)")
        
        return df
    
    def create_readmission_outcome(self) -> Optional[pd.DataFrame]:
        """Create 180-day readmission outcome"""
        logger.info("Creating readmission outcome...")
        
        # Extract 6-month follow-up data
        df_6month = self.df[self.df["redcap_event_name"].str.contains("6month")]
        
        # Get readmission columns
        readmission_cols = ['record_id'] + [f'readmission_date{"_" + str(i) if i > 1 else ""}' 
                                          for i in range(1, 6)]
        
        # Check for missing columns
        missing_cols = [col for col in readmission_cols if col not in df_6month.columns]
        if missing_cols:
            logger.warning(f"Missing readmission columns: {missing_cols}")
            readmission_cols = [col for col in readmission_cols if col in df_6month.columns]
        
        df_6month_readmission = df_6month[readmission_cols]
        
        # Get initial hospitalization data
        df_index = self.df[self.df["redcap_event_name"].str.contains("index")]
        if 'date_of_hospital_admission' not in df_index.columns:
            logger.error("Missing date_of_hospital_admission column")
            return None
        
        df_index = df_index[['record_id', 'date_of_hospital_admission']]
        
        # Merge data
        df_hospital = pd.merge(df_index, df_6month_readmission, on="record_id", how="left")
        
        # Calculate readmission times
        df_hospital = self._calculate_readmission_times(df_hospital)
        
        # Create 180-day readmission indicator
        df_hospital = self._create_readmission_indicator(df_hospital)
        
        return df_hospital[['record_id', '180_readmission']]
    
    def _calculate_readmission_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time differences between admissions"""
        logger.info("Calculating readmission times...")
        
        try:
            for i in range(1, 6):
                col_suffix = '' if i == 1 else f'_{i}'
                date_col = f'readmission_date{col_suffix}'
                
                if date_col in df.columns:
                    time_col = f'readmission_time{col_suffix}'
                    df[time_col] = pd.to_datetime(df[date_col], format='%d/%m/%y %H:%M', errors='coerce') - \
                                 pd.to_datetime(df['date_of_hospital_admission'], format='%d/%m/%y %H:%M', errors='coerce')
                    
                    valid_dates = df[time_col].notna().sum()
                    invalid_dates = df[date_col].notna().sum() - valid_dates
                    if invalid_dates > 0:
                        logger.warning(f"{date_col} has {invalid_dates} invalid dates")
                    
                    logger.info(f"Calculated {valid_dates} {time_col}")
        except Exception as e:
            logger.error(f"Error calculating readmission times: {str(e)}")
        
        return df
    
    def _create_readmission_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create 180-day readmission indicator"""
        logger.info("Creating readmission indicator...")
        
        readmission_time_cols = [col for col in df.columns if 'readmission_time' in col]
        
        if readmission_time_cols:
            df['180_readmission'] = (
                df[readmission_time_cols]
                .apply(lambda x: x.le(pd.Timedelta(days=180)) & x.notna(), axis=0)
                .any(axis=1)
            ).astype(int)
            
            readmission_count = df['180_readmission'].sum()
            total_count = len(df)
            readmission_rate = readmission_count / total_count * 100
            logger.info(f"180-day readmission count: {readmission_count}/{total_count} ({readmission_rate:.2f}%)")
        else:
            logger.warning("No readmission time columns found, setting 180_readmission to 0")
            df['180_readmission'] = 0
        
        return df
    
    def process(self) -> pd.DataFrame:
        """Main processing pipeline"""
        # Load data
        self.df = self.load_data()
        
        # Extract and merge timepoint data
        index_df, m3_df, m6_df = self.extract_timepoint_data()
        self.df_whodas = index_df.merge(m3_df, on='record_id', how='left')
        self.df_whodas = self.df_whodas.merge(m6_df, on='record_id', how='left')
        
        # Calculate scores
        base_cols = [f's{i}' for i in range(1, 13)]
        cols_90 = [f's{i}_90' for i in range(1, 13)]
        cols_180 = [f's{i}_180' for i in range(1, 13)]
        
        self.df_whodas['score1'] = self.df_whodas[base_cols].sum(axis=1)
        self.df_whodas['score1_90'] = self.df_whodas[cols_90].sum(axis=1)
        self.df_whodas['score1_180'] = self.df_whodas[cols_180].sum(axis=1)
        
        # Handle NaN scores
        for cols, score_col in [(base_cols, 'score1'), (cols_90, 'score1_90'), (cols_180, 'score1_180')]:
            self.df_whodas.loc[self.df_whodas[cols].isna().all(axis=1), score_col] = np.nan
        
        # Extract key columns
        self.df_whodas = self.df_whodas[['record_id', 'score1', 'score1_90', 'score1_180']]
        
        # Add hospital_los data
        hospital_los_data = self.df[self.df["redcap_event_name"].str.contains("index")][['record_id', 'hospital_los']]
        self.df_whodas = pd.merge(self.df_whodas, hospital_los_data, on='record_id', how='left')
        
        
        # Add death data
        death_data = self.df[self.df["redcap_event_name"].str.contains("index")][['record_id', 'days_death_surgery']]
        self.df_whodas = pd.merge(self.df_whodas, death_data, on='record_id', how='left')
        
        # Scale and fill scores
        df_scaled = self.scale_and_fill_scores(self.df_whodas)
        
        # Create outcome variables
        self.final_df = self.create_outcome_variables(df_scaled)
        
        # Add readmission data
        readmission_df = self.create_readmission_outcome()
        if readmission_df is not None:
            self.final_df = pd.merge(self.final_df, readmission_df, on='record_id', how='left')
            self.final_df['180_readmission'] = self.final_df['180_readmission'].fillna(0).astype(int)
            logger.info("Successfully merged readmission data")
        else:
            logger.warning("Could not create readmission data, adding empty 180_readmission column")
            self.final_df['180_readmission'] = 0
        
        # Final checks and statistics
        self._final_checks()
        self._log_statistics()
        
        # Save results
        self._save_results()
        
        return self.final_df
    
    def _final_checks(self):
        """Perform final data quality checks"""
        if 'score1' in self.final_df.columns and self.final_df['score1'].isna().any():
            missing_count = self.final_df['score1'].isna().sum()
            logger.warning(f"Found {missing_count} missing score1 values, filling with 0")
            self.final_df['score1'] = self.final_df['score1'].fillna(0)
        else:
            logger.info("No missing values in score1 column")
    
    def _log_statistics(self):
        """Log final statistics"""
        logger.info(f"Total records: {len(self.final_df)}")
        logger.info(f"3-month adverse outcome count: {self.final_df['dd_3month'].sum()} "
                   f"({self.final_df['dd_3month'].sum()/len(self.final_df)*100:.2f}%)")
        logger.info(f"6-month adverse outcome count: {self.final_df['dd_6month'].sum()} "
                   f"({self.final_df['dd_6month'].sum()/len(self.final_df)*100:.2f}%)")
        logger.info(f"Extended LOS count: {self.final_df['los_target'].sum()} "
                   f"({self.final_df['los_target'].sum()/len(self.final_df)*100:.2f}%)")
        logger.info(f"180-day readmission count: {self.final_df['180_readmission'].sum()} "
                   f"({self.final_df['180_readmission'].sum()/len(self.final_df)*100:.2f}%)")
    
    def _save_results(self):
        """Save results to CSV"""
        logger.info(f"Saving results to {self.config.output_file}...")
        self.final_df.to_csv(self.config.output_file, index=False)
        logger.info("Processing completed!")

def main(input_file: str = './MIDASCDWH_DATA_2024-11-12_1309.csv', 
         output_file: str = './whodas_outcomes.csv') -> pd.DataFrame:
    """Main entry point for the program"""
    config = Config(input_file=input_file, output_file=output_file)
    processor = WHODASProcessor(config)
    return processor.process()

if __name__ == "__main__":
    main()