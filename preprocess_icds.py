import pandas as pd 
import numpy as np
import os
from utils import *
from datetime import datetime

class IcdsPreprocessing:
    def __init__(self):
        self.icd_pac_columns = {
            "atrial_fibrillation":{
                "pac":["af_pac"],
                # "pac":[],
                "icd":["atrial_fibrillation_icd"],
            },
            "cardiac_transplant":{
                "pac":["cardiac_transplant_pac"],
                "icd":["cardiac_transplant_icd"],
            },
            "pvd":{
                "pac":["pvd_pac"],
                "icd":["pvd_icd"],
            },
            "other_cardiac":{
                "pac":["other_cardiac_pac"],
                "icd":["chronic_cardiac_icu"],
            },
            "asthma":{
                "pac":["asthma_pac"],
                "icd":["asthma_icd"],
            },
            "copd": {
                "pac":["copd_pac"],
                "icd":["copd_icd"]
            },
            "osa":{
                "pac":["osa_pac"],
                "icd":["osa_icd"]   
            },
            "pht":{
                "pac":["pht_pac"],
                "icd":["pht_icd"]
            },
            "lung_transplant":{
                "pac":["lung_transplant_pac"],
                "icd":["lung_transplant_icd"]
            },
            "smoking":{
                "pac":["current_smoker_pac", "ex_smoker_pac"],
                "icd":["smoking_icd"]
            },
            "other_respiratory":{
                "pac":["other_respiratory_pac"],
                "icd":["other_respiratory_icd"]   
            },
            "cerebrovascular":{
                "pac":["cerebrovascular_pac"],
                "icd":["cerebrovascular_icd"]
            },
            "dementia":{
                "pac":["dementia_pac"],
                "icd":["dementia_icd"]   
            },
            "hemiplegia":{
                "pac":["hemiplegia_pac"],
                "icd":["hemiplegia_icd"]   
            },
            "endocrine":{
                "pac":["endocrine_pac"],
                "icd":["endocrine_icd"]   
            },
            "type1_diabetes":{
                "icd":["type1_with_icd", "type1_without_icd"],
                "pac":["type1_pac"]
            },
            # "type2_diabetes":{
            #     "icd":["type2_with_icd", "type2_without_icd", "type2_insulin_icd"],
            #     "pac":[
            #         "type2_pac"
            #     ]
            # },
            "metabolic":{
                "pac":["metabolic_pac"],
                "icd":["metabolic_icd"]   
            },
            "ckd":{
                "pac":["ckd_pac"],
                "icd":["ckd_icd"]   
            },
            "cld":{
                "pac":["cld_pac"],
                "icd":["cld_icd"]   
            },
            "haematological":{
                "pac":["haematological_pac"],
                "icd":["haematological_icd"]   
            },
            "hiv_aids":{
                "pac":["hiv_pac"],
                "icd":["hiv_aids_icd"]   
            },
            'rheumatoid':{
                "pac":["rheumatoid_arthritis"],
                "icd":["rheumatoid_ct_icd"]
            },
            "osteoarthritis":{
                "pac":["osteoarthritis_pac"],
                "icd":["osteoarthritis_osteoporosis_icd"]   
            },
            "gord":{
                "pac":["gord_pac"],
                "icd":["gord_icd"]   
            },
            "hepatitis":{
                "icd":["infectious_hepatitis_icd"],
                "pac":["hepatitis_b_pac", "hepatitis_c_pac"]
            },
            "persistent_pain_3_months":{
                "icd":[],
                "pac":["persistent_pain_3_months"]
            },
            "opioids_120mg_oral":{
                "icd":[],
                "pac":["opioids_120mg_oral"]
            },
            "anti_neuropathic_meds":{
                "icd":[],
                "pac":["anti_neuropathic_meds"]
            },
            "operation_painful_site":{
                "icd":[],
                "pac":["operation_painful_site"]
            },
            "chronic_pain":{
                "pac":[],
                "icd":["chronic_pain_icd"]
            },
            "pain_specialist":{
                "pac":["pain_specialist"],
                "icd":[]
            },
            "stroke":{
                "pac":["stroke"],
                "icd":[]
            },
            "tia":{
                "pac":["tia"],
                "icd":[]
            },
            "neurological_condition":{
                "pac":["neurological_condition_pac"],
                "icd":[]
            },
            "epilepsy":{
                "pac":["epilepsy"],
                "icd":[]
            },
            "psychiatric_illness":{
                "pac":["psychiatric_illness"],
                "icd":[]
            },
            "neuromuscular_disorder":{
                "pac":["neuromuscular_disorder"],
                "icd":[]
            },
            "intellectual_development":{
                "pac":[],
                "icd":["intellectual_development"]
            },
            "degenerative":{
                "pac":[],
                "icd":["degenerative_icd"]
            },
            "congenital":{
                "pac":[],
                "icd":["congenital_icd"]
            },
            "organic_brain":{
                "pac":[],
                "icd":["organic_brain_icd"]
            },
            "psych":{
                "pac":[],
                "icd":["psych_icd"]
            },
            "epilepsy":{
                "pac":[],
                "icd":["epilepsy_icd"]
            },
            "hemiplegia":{
                "pac":["hemiplegia_pac"],
                "icd":["hemiplegia_icd"]
            },
            "myocardial_ischaemia":{
                "icd":["myocardial_ischaemia_icd"],
                "pac":[]
            },
            "myocardial_infarction": {
                "icd":["myocardial_infarction_icd"],
                "pac":[]
            },
            "anaemia":{
                "icd":[],
                "pac":["anaemia"]
            },
            "bleeding_disorder":{
                "icd":[],
                "pac":["bleeding_disorder"]
            },
            "thromboembolic_disease":{
                "icd":[],
                "pac":["thromboembolic_disease"]
            },
            "haem_ca":{
                "icd":["haem_ca_icd"],
                "pac":["haem_ca_pac"]
            },
            "lymphoma_icu":{
                "icd":[],
                "pac":["lymphoma_icu"]
            },
            "leukaemia_myeloma_icu":{
                "icd":[],
                "pac":["leukaemia_myeloma_icu"]
            },
            "ischaemic_heart_disease":{
                "icd":["ischaemic_heart_disease_icd"],
                "pac":["ischaemic_heart_disease"]
            },
            "heart_failure":{
                "icd":[],
                "pac":[]
            },
            "heart_failure":{
                "icd":["heart_failur_icd_10"],
                "pac":["heart_failure_pac", "heart_failure_pac_powerfor"]
            },
            "valvular_heart":{
                "icd":["valvular_heart_disease_icd"],
                "pac":["valvular_heart_pac", "valvular_heart_disease_pac"]
            },
            "living_alone":{
                "icd":["living_alone_icd"],
                "pac":[]
            },
            "acute_renal_failure":{
                "icd":["acute_renal_failure_icd_10"],
                "pac":[]
            },
            "hypertension":{
                "icd":["hypertension_icd"],
                "pac":["hypertension_pac"]
            },
            "dialysis":{
                "icd":["dialysis_icd"],
                "pac":["haemodialysis", "peritoneal_dialysis"]
            },
            "cancer":{
                "icd":["cancer_primary_icd", "cancer_secondary_icd"],
                "pac":["cancer_unspecified_pac", "cancer_localised_pac", "cancer_metastatic_pac"]
            },
            "gastrointestinal":{
                "icd":["gastrointestinal_icd"],
                "pac":[]
            }
            
        }
    
    def load_dataset(self, csv_filename):
        # Check if file exists
        if not os.path.exists(csv_filename):
            raise FileNotFoundError(f"CSV Data file not found: {csv_filename}")
        
        # Load the original dataset
        print("Loading the dataset...")
        df = pd.read_csv(csv_filename, low_memory=False)

        # Filter for index events
        df = df[df["redcap_event_name"].str.contains("index")]
        print(f"Dataset loaded successfully for index events: {df.shape}")
        return df

    def check_disease_present(self, df, disease_name, pac_columns, icd_columns):
        """
        Mark rows as 1 if disease is present based on any PAC or ICD columns, otherwise 0.
        
        Parameters:
        - df: The pandas DataFrame.
        - pac_columns: A list of column names representing PAC (binary).
        - icd_columns: A list of column names representing ICD codes.
        
        Returns:
        - A pandas DataFrame with a new column '[disease_name]_disease_present'.
        """
        if isinstance(pac_columns, str):
            pac_columns = [pac_columns]
        if isinstance(icd_columns, str):
            icd_columns = [icd_columns]
        def is_disease_present(row):
            pac_positive = any(row[col] == 1 for col in pac_columns)
            icd_positive = any(not pd.isnull(row[col]) and row[col] != '' for col in icd_columns)
            return 1 if pac_positive or icd_positive else 0
        
        df[disease_name + '_disease_present'] = df.apply(is_disease_present, axis=1)
        return df

    def filter_low_variance_columns(self, df):
        print("Fitering zero variance columns...")
        print(f"Columns before applying operation are {len(df.columns)}")
        
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            print("Dropped columns:", constant_cols)
            for col in constant_cols:
                unique_vals = df[col].unique()
                print(f" - {col}: unique value(s) = {unique_vals}")
            df = df.drop(columns=constant_cols)
            print(f"Columns after filtering zero variance column is {len(df.columns)}")
        else:
            print("No column with zero variance")
          
        filtered_columns = self.calc_non_low_variance_features(df)
        print(f"Columns after filtering low variance columns are {len(filtered_columns)}")
        
        return df[filtered_columns]

    def calc_non_low_variance_features(self, df, exclude_na=True, top_feature_prop_threshold=19):
        df_clean = df.copy()
        low_var_features = pd.DataFrame()

        # Step 1: Identify columns with only one unique value
        low_var_features['less_than_two'] = df.apply(lambda x: x.nunique(dropna=exclude_na) < 2, axis=0)
        
        # Ensure the indices are aligned
        low_var_features.index = df_clean.columns

        # Step 2: Identify columns with imbalanced distributions (dominant value too frequent)
        columns_after_one_unique = df_clean.columns[~low_var_features['less_than_two']]

        # Step 3: Calculate the value ratio for each column
        low_var_features['val_count_ratio'] = df_clean[columns_after_one_unique].apply(
            lambda x: self._calculate_value_ratio(x, exclude_na), axis=0
        )

        # Step 4: Filter out columns with a high dominant value ratio
        valid_columns = low_var_features[low_var_features['val_count_ratio'] <= top_feature_prop_threshold].index
        final_columns = df_clean[valid_columns].columns

        print(f"Columns remaining after filtering: {len(final_columns)}")
        print(f"Columns dropped due to low variance: {list(set(df_clean.columns) - set(final_columns))}")

        return final_columns

    def _calculate_value_ratio(self, column, exclude_na=True):
        """
        Calculates the ratio of the most frequent value's count to the second most frequent value's count.
        If there are fewer than two unique values, returns a high value (so it is considered low variance).
        
        Parameters:
        - column: pd.Series, the column to calculate value counts for.
        - exclude_na: bool, whether to exclude NaN values.
        
        Returns:
        - The ratio of the most frequent value to the second most frequent value.
        """
        value_counts = column.value_counts(dropna=exclude_na)
        if len(value_counts) < 2:
            return float('inf')  # Return a very high ratio for columns with one unique value
        most_frequent = value_counts.iloc[0]
        second_most_frequent = value_counts.iloc[1]
        return most_frequent / second_most_frequent
    
    def preprocess_icd_data(self, df):
        print("\n........Preprocessing Icd Columns")
        for key in self.icd_pac_columns:
            disease_name = key
            val = self.icd_pac_columns[key]
            df = self.check_disease_present(df, disease_name, val['pac'], val['icd'])

            disease_count = df[disease_name+'_disease_present'].sum()
            # Print the count
            print(f"Number of rows with {disease_name} disease present: {disease_count}")
        final_columns = self.calc_non_low_variance_features(df)
        final_columns = final_columns.tolist() 
        df = df[final_columns]
        return df
    
    def analyze_feature_importance(X, y, task='classification', method='mutual_info', top_n=10):
        """
        Analyze and return feature importance using specified method.

        Parameters:
            X (pd.DataFrame): Feature matrix
            y (pd.Series or np.array): Target vector
            task (str): 'classification' or 'regression'
            method (str): 'mutual_info', 'random_forest', or 'linear_model'
            top_n (int): Number of top features to return

        Returns:
            pd.DataFrame: Feature importance scores sorted by importance
        """

        if task not in ['classification', 'regression']:
            raise ValueError("task must be 'classification' or 'regression'")

        if method == 'mutual_info':
            if task == 'classification':
                scores = mutual_info_classif(X, y, discrete_features='auto')
            else:
                scores = mutual_info_regression(X, y)
            importance_df = pd.DataFrame({'feature': X.columns, 'importance': scores})

        elif method == 'random_forest':
            model = RandomForestClassifier() if task == 'classification' else RandomForestRegressor()
            model.fit(X, y)
            importance_df = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})

        elif method == 'linear_model':
            model = LogisticRegression(max_iter=1000) if task == 'classification' else LinearRegression()
            model.fit(X, y)
            coefs = model.coef_[0] if task == 'classification' else model.coef_
            importance_df = pd.DataFrame({'feature': X.columns, 'importance': np.abs(coefs)})

        else:
            raise ValueError("method must be one of: 'mutual_info', 'random_forest', 'linear_model'")

        return importance_df.sort_values(by='importance', ascending=False).head(top_n)

    
def main():
    # Load data
    csv_filename = "MIDASCDWH_DATA_2024-11-12_1309.csv"
    output_file = "MIDAS_preprocessed_icds_" + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.csv'
    
    icdpreprocessor = IcdsPreprocessing()
    df = icdpreprocessor.load_dataset(csv_filename)
    df = icdpreprocessor.preprocess_icd_data(df)
    df = icdpreprocessor.filter_low_variance_columns(df)
    print(f"Columns after filtering low variance columns are {df.columns}")
    
    # Save the processed data
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main() 