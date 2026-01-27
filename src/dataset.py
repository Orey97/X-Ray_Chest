import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

class Dataset:

    def __init__(self, data_path):
        # STRATEGY CHANGE: Explicit path handling only.
        # No assumptions about project root or relative locations.
        self.file_path = os.path.abspath(data_path)
        
        print(f"[DATASET] Resolved Absolute Path: {self.file_path}")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CRITICAL ERROR: The provided path does not exist on this filesystem:\n{self.file_path}")
            
        self.data = None 
        self.load_data()

    def load_data(self):
        """
        Loads the CSV file into a pandas DataFrame.
        This provides the raw manifest of all available X-ray images and their metadata.
        """
        print(f"[DATASET] Loading data from: {self.file_path}")
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def export_data(self, file_name):
        if self.data is None:
            print("Data not loaded. Please load the data first.")
        else:
            print("Exporting DataFrame")
            self.data.to_csv(file_name, index=False)

    def visualize_dataset(self, type=None, n_cols=5):
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return
        if type is None:
            type = ["head", "info", "columns"]
        for t in type:
            if t == "head":
                print(self.data.head(n=n_cols))
            if t == "info":
                print(self.data.describe(include="all"))
            if t == "columns":
                print(self.data.columns)

    def drop_columns(self, columns):
        if self.data is None:
            print("Data not loaded")
            return
        print(f'---------------------- Dropping columns {columns} --------------------------')
        self.data = self.data.drop(columns=columns)    

    def modify_column_name(self, old_name, new_name):
        if self.data is None:
            print("Data not loaded")
            return
        print(f'---------------------- Modifying column name from {old_name} to {new_name} --------------------------')
        self.data = self.data.rename(columns={old_name: new_name})

    # Reformats the age column from '045Y' string format to integer 45
    def modify_age_column(self, column_name):
        """
        Cleans the Age column.
        The dataset format is often '034Y' (string), which cannot be used for numerical analysis.
        We strip the 'Y' and convert it to an integer.
        """
        if self.data is None:
            print("Data not loaded")
            return
        print(f'---------------------- Modifying age column {column_name} --------------------------')
        self.data[column_name] = self.data[column_name].str.replace('Y','').astype(int)


    def enconder_gender(self, column_name="PatientGender"):
        if self.data is None:
            print("Data not loaded")
            return
        print(f'----------------Encoding gender column--------------------------')
        mapping = {'M': 0, 'F': 1}
        self.data[column_name] = self.data[column_name].map(mapping).astype(float)

    # Standardizes column names to lowercase and removes special characters
    def clean_column_names(self):
        if self.data is None:
            print("Data not loaded")
            return
        print("---------------------- Cleaning column names --------------------------")
        
        # 1. Basic Cleaning
        self.data.columns = (
            self.data.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("(", "")
                .str.replace(")", "")
        )
        
        # 2. Canonical Standardization
        # We want internal code to always use 'image', 'patientid'
        rename_map = {}
        if "image_index" in self.data.columns:
            rename_map["image_index"] = "image"
        if "patient_id" in self.data.columns:
            rename_map["patient_id"] = "patientid"
        if "finding_labels" in self.data.columns:
             # Just ensures we can find it later, even if we renamed it earlier.
             pass 
             
        if rename_map:
            print(f"Standardizing columns: {rename_map}")
            self.data = self.data.rename(columns=rename_map)

    # Parses and cleans multi-label entries (separated by pipe |)
    def clean_labels(self, column_name):
        """
        Parses the Multi-Label column.
        Raw labels are in the format "ConditionA|ConditionB".
        This function splits them into a list ["ConditionA", "ConditionB"],
        preparing them for One-Hot Encoding.
        """
        if self.data is None:
            print("Data not loaded")
            return
        print(f'---------------------- Cleaning labels in column {column_name} --------------------------')
        self.data[column_name] = self.data[column_name].str.replace(" ","").str.split('|')

    # Performs One-Hot Encoding on the labels
    def one_hot_encode_labels(self, column="label", explicit_labels=None):
        """
        Converts the list of text labels into a mathematical vector representation.
        CRITICAL: If 'explicit_labels' is provided (from Schema), it forces that exact order.
        This prevents 'Data Leakage' where train and test sets might have different sorted orders
        if a rare class is missing in one of them.
        """
        if explicit_labels:
            print(f"[DATASET] One-Hot Encoding utilizing STRICT SCHEMA with {len(explicit_labels)} classes.")
            unique_labels = explicit_labels
        else:
            print("[DATASET] WARNING: No explicit header provided. Deriving from data (Risk of Skew).")
            unique_labels = sorted(
                set(label for sublist in self.data[column] for label in sublist)
            )
            
        for label in unique_labels:
            # Check if label exists in the row's list
            self.data[label] = self.data[column].apply(lambda x: int(label in x))

    # For CNN training, we only need the image path, patient ID, and the one-hot encoded labels
    def select_relevant_columns(self):
        # Columns to explicitly exclude (metadata that should not be targets)
        exclude_cols = [
            'label', 
            'finding_labels',
            'follow-up', 
            'patientage', 
            'patientgender', 
            'viewposition', 
            'originalimagewidth', 
            'originalimageheight', 
            'originalimagepixelspacingx', 
            'originalimagepixelspacingy',
            'follow-up_#',
            'patient_age',
            'patient_gender',
            'view_position',
            'original_image_width',
            'original_image_height',
            'original_image_pixel_spacing_x',
            'original_image_pixel_spacing_y'
        ]
        
        # Ensure we keep image and patientid if they exist
        base_cols = ['image', 'patientid']
        targets = [
            col for col in self.data.columns 
            if col not in base_cols and col not in exclude_cols
        ]
        
        print(f"Keeping columns (Targets): {targets}")
        self.data = self.data[base_cols + targets]


    # Verify that image files exist in the specified directory to avoid training crashes due to missing/corrupt files
    def check_image_files(self, image_dir):
        """
        Data Integrity Check.
        Verifies that all images in the dataframe exist on disk.
        Returns a list of missing filenames.
        """
        abs_image_dir = os.path.abspath(image_dir)
        print(f"[DATA INTEGRITY] Checking images in Absolute Path: {abs_image_dir}")
        
        if not os.path.isdir(abs_image_dir):
             raise FileNotFoundError(f"CRITICAL ERROR: The image directory provided does not exist:\n{abs_image_dir}")

        # Column name might be 'image' or 'image_index' depending on cleaning
        col = 'image' if 'image' in self.data.columns else 'image_index'
        if col not in self.data.columns:
            print("[DATA] Warning: Could not find image column to check file existence.")
            return []
            
        missing = [img for img in self.data[col] if not os.path.exists(os.path.join(abs_image_dir, img))]
        return missing
    
    # IMPORTANT: Risk of DATA LEAKAGE if splitting is done directly on the dataframe (random shuffle).
    # The same patient may have multiple images (follow-ups) that look very similar.
    # If we don't split by patient, the model might learn to recognize the patient anatomy instead of the pathology.
    # We must ensure distinct patients across Train, Validation, and Test sets.
    def patient_split(self, test_size=0.2, val_size=0.125): # Note: val_size applies to the remaining training data 
        """
        Splits the dataset ensuring ZERO PATIENT LEAKAGE.
        Problem: A single patient often has multiple X-rays (follow-ups).
        If we split randomly by image, patient X could have image A in Train and image B in Test.
        The model would memorize Patient X's anatomy instead of learning the pathology.
        
        Solution: We split by 'PatientID'. All images from a unique patient go strictly into ONE set (Train OR Val OR Test).
        """
        if self.data is None:
            print("Data not loaded")
            return
        print('---------------------- Splitting dataset by PatientID --------------------------')
        patients = self.data['patientid'].unique()
        train_p, test_p = train_test_split(patients, test_size=test_size, random_state=42)
        train_p, val_p = train_test_split(train_p, test_size=val_size, random_state=42)
        train_df = self.data[self.data['patientid'].isin(train_p)]
        val_df = self.data[self.data['patientid'].isin(val_p)]
        test_df = self.data[self.data['patientid'].isin(test_p)]
        return train_df, val_df, test_df
