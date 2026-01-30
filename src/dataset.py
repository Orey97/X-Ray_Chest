"""
=============================================================================
                    DATASET.PY - Data Management Pipeline
=============================================================================

PURPOSE:
    Handles all data preprocessing from raw CSV to training-ready format.
    Implements PATIENT-AWARE splitting to prevent data leakage.

PIPELINE STAGES:
    
    1. LOAD: Read CSV metadata
    2. CLEAN: Standardize column names
    3. PARSE: Split multi-label strings ("Pneumonia|Edema")
    4. ENCODE: Convert to one-hot vectors
    5. VALIDATE: Check image files exist
    6. SPLIT: Separate into train/val/test (BY PATIENT)

THE DATA LEAKAGE PROBLEM:

    ❌ WRONG: Random Split by Image
    
        Patient A has 5 chest X-rays (taken over 2 years of monitoring)
        Random split might put:
          - Image 1, 3, 5 → Training
          - Image 2, 4 → Testing
        
        PROBLEM: The images are very similar (same patient's anatomy).
        Model memorizes Patient A's unique rib shape, heart position, etc.
        Test accuracy is artificially HIGH because it's recognizing the patient,
        not the disease!
        
    ✅ CORRECT: Split by Patient ID
    
        All of Patient A's images → Training (or all → Testing)
        Model CANNOT cheat by recognizing patient anatomy
        Test accuracy reflects TRUE generalization to NEW patients

=============================================================================
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


class Dataset:
    """
    Comprehensive data management class for chest X-ray classification.
    
    This class handles the full data preparation pipeline:
    1. Loading CSV metadata
    2. Cleaning and standardizing column names
    3. Parsing multi-label disease annotations
    4. One-hot encoding for neural network consumption
    5. Patient-aware train/val/test splitting
    
    Attributes:
        file_path (str): Absolute path to the CSV file
        data (DataFrame): The pandas DataFrame containing all records
    """

    def __init__(self, data_path):
        """
        Initialize the dataset manager.
        
        Args:
            data_path (str): Path to the CSV metadata file.
                            Can be relative or absolute.
                            
        Note:
            Immediately loads the CSV into memory upon initialization.
            For very large datasets (>100K rows), this may take a few seconds.
        """
        # ALWAYS convert to absolute path for reliability
        # Relative paths cause issues when scripts are run from different directories
        self.file_path = os.path.abspath(data_path)
        
        print(f"[DATASET] Resolved Absolute Path: {self.file_path}")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CRITICAL ERROR: The provided path does not exist on this filesystem:\n{self.file_path}")
            
        self.data = None 
        self.load_data()

    def load_data(self):
        """
        Load the CSV file into a pandas DataFrame.
        
        The NIH Chest X-ray dataset CSV contains columns like:
        - Image Index: filename (e.g., "00000001_000.png")
        - Finding Labels: pipe-separated diseases ("Pneumonia|Effusion")
        - Patient ID: unique patient identifier
        - Follow-up #: which visit this is (patients have multiple scans)
        - Patient Age: age at time of scan
        - Patient Gender: M or F
        - View Position: PA (front) or AP (portable)
        """
        print(f"[DATASET] Loading data from: {self.file_path}")
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def export_data(self, file_name):
        """Export the processed DataFrame to a new CSV file."""
        if self.data is None:
            print("Data not loaded. Please load the data first.")
        else:
            print("Exporting DataFrame")
            self.data.to_csv(file_name, index=False)

    def visualize_dataset(self, type=None, n_cols=5):
        """
        Display dataset information for debugging.
        
        Args:
            type (list): Types of info to show: ["head", "info", "columns"]
            n_cols (int): Number of rows for head display
        """
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
        """Remove specified columns from the DataFrame."""
        if self.data is None:
            print("Data not loaded")
            return
        print(f'---------------------- Dropping columns {columns} --------------------------')
        self.data = self.data.drop(columns=columns)    

    def modify_column_name(self, old_name, new_name):
        """Rename a column."""
        if self.data is None:
            print("Data not loaded")
            return
        print(f'---------------------- Modifying column name from {old_name} to {new_name} --------------------------')
        self.data = self.data.rename(columns={old_name: new_name})

    def modify_age_column(self, column_name):
        """
        Convert age from string format to integer.
        
        NIH dataset stores age as "045Y" (string with 3 digits + 'Y').
        We need it as integer 45 for potential use in demographic analysis.
        """
        if self.data is None:
            print("Data not loaded")
            return
        print(f'---------------------- Modifying age column {column_name} --------------------------')
        self.data[column_name] = self.data[column_name].str.replace('Y','').astype(int)


    def enconder_gender(self, column_name="PatientGender"):
        """Encode gender as binary: M=0, F=1."""
        if self.data is None:
            print("Data not loaded")
            return
        print(f'----------------Encoding gender column--------------------------')
        mapping = {'M': 0, 'F': 1}
        self.data[column_name] = self.data[column_name].map(mapping).astype(float)

    def clean_column_names(self):
        """
        Standardize column names for consistent access.
        
        Transformations:
        - Strip whitespace
        - Convert to lowercase
        - Replace spaces with underscores
        - Remove parentheses
        
        Also renames common variations:
        - "Image Index" → "image"
        - "Patient ID" → "patientid"
        
        This ensures the rest of the code can reliably use
        df["image"] and df["patientid"] without worrying about
        original column naming conventions.
        """
        if self.data is None:
            print("Data not loaded")
            return
        print("---------------------- Cleaning column names --------------------------")
        
        # Step 1: Basic cleaning
        self.data.columns = (
            self.data.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("(", "")
                .str.replace(")", "")
        )
        
        # Step 2: Canonical standardization
        # Different datasets might use slightly different names
        # We map them all to our consistent internal naming
        rename_map = {}
        if "image_index" in self.data.columns:
            rename_map["image_index"] = "image"
        if "patient_id" in self.data.columns:
            rename_map["patient_id"] = "patientid"
             
        if rename_map:
            print(f"Standardizing columns: {rename_map}")
            self.data = self.data.rename(columns=rename_map)

    def clean_labels(self, column_name):
        """
        Parse multi-label strings into lists.
        
        The NIH dataset stores labels as pipe-separated strings:
        - "Pneumonia|Effusion" 
        - "Cardiomegaly"
        - "No Finding"
        
        This function converts them to Python lists:
        - ["Pneumonia", "Effusion"]
        - ["Cardiomegaly"]
        - ["No Finding"]
        
        This prepares the data for one-hot encoding.
        """
        if self.data is None:
            print("Data not loaded")
            return
        print(f'---------------------- Cleaning labels in column {column_name} --------------------------')
        self.data[column_name] = self.data[column_name].str.replace(" ","").str.split('|')

    def one_hot_encode_labels(self, column="label", explicit_labels=None):
        """
        Convert label lists to one-hot encoded columns.
        
        CRITICAL: If explicit_labels is provided from schema, we use THAT
        exact order. This ensures consistency between training and inference.
        
        Input (column value): ["Pneumonia", "Effusion"]
        
        Output (new columns):
            Atelectasis: 0
            Cardiomegaly: 0
            Effusion: 1      ← Has this disease
            Pneumonia: 1     ← Has this disease
            ...
        
        Args:
            column (str): Name of the column containing label lists
            explicit_labels (list): If provided, use this exact label set and order.
                                   This comes from schema.json for strict consistency.
                                   
        Why explicit_labels matters:
            Without it, we derive labels from the data using sorted(set(...)).
            But if train/test splits have different disease distributions,
            the label order could differ! This causes silent prediction errors.
        """
        if explicit_labels:
            print(f"[DATASET] One-Hot Encoding utilizing STRICT SCHEMA with {len(explicit_labels)} classes.")
            unique_labels = explicit_labels
        else:
            print("[DATASET] WARNING: No explicit header provided. Deriving from data (Risk of Skew).")
            unique_labels = sorted(
                set(label for sublist in self.data[column] for label in sublist)
            )
            
        # Create a new column for each label
        for label in unique_labels:
            # For each row, check if 'label' is in that row's list of labels
            # Result is 1 if present, 0 if not
            self.data[label] = self.data[column].apply(lambda x: int(label in x))

    def select_relevant_columns(self):
        """
        Keep only columns needed for training.
        
        After one-hot encoding, DataFrame has:
        - Metadata columns (age, gender, view position, etc.)
        - Image path column
        - Patient ID column  
        - One-hot label columns (14 pathologies)
        
        For CNN training, we ONLY need:
        - image: path to the image file
        - patientid: for splitting (removed after split)
        - [label columns]: the 14 one-hot encoded targets
        
        We DROP all metadata columns as they're not used in image classification.
        """
        # Columns to explicitly exclude (metadata, not targets)
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
        
        # Always keep image path and patient ID
        base_cols = ['image', 'patientid']
        
        # Everything else (the one-hot labels) becomes our targets
        targets = [
            col for col in self.data.columns 
            if col not in base_cols and col not in exclude_cols
        ]
        
        print(f"Keeping columns (Targets): {targets}")
        self.data = self.data[base_cols + targets]


    def check_image_files(self, image_dir):
        """
        Verify that all referenced images actually exist on disk.
        
        This is a DATA INTEGRITY check. If the CSV references images
        that don't exist (deleted, corrupted, wrong path), training
        will crash mid-epoch with confusing errors.
        
        Better to detect missing files BEFORE training starts.
        
        Args:
            image_dir (str): Directory containing the image files
            
        Returns:
            list: Filenames that are in CSV but not on disk
        """
        abs_image_dir = os.path.abspath(image_dir)
        print(f"[DATA INTEGRITY] Checking images in Absolute Path: {abs_image_dir}")
        
        if not os.path.isdir(abs_image_dir):
             raise FileNotFoundError(f"CRITICAL ERROR: The image directory provided does not exist:\n{abs_image_dir}")

        # Handle both possible column names
        col = 'image' if 'image' in self.data.columns else 'image_index'
        if col not in self.data.columns:
            print("[DATA] Warning: Could not find image column to check file existence.")
            return []
            
        missing = [img for img in self.data[col] if not os.path.exists(os.path.join(abs_image_dir, img))]
        return missing
    
    def patient_split(self, test_size=0.2, val_size=0.125):
        """
        Split dataset ensuring ZERO PATIENT LEAKAGE.
        
        ═══════════════════════════════════════════════════════════════════
        THE CRITICAL INSIGHT:
        ═══════════════════════════════════════════════════════════════════
        
        A single patient often has MULTIPLE X-rays taken over time:
        - Initial diagnosis scan
        - Follow-up after treatment
        - Annual monitoring scans
        
        These images are HIGHLY SIMILAR (same person's anatomy).
        
        If we split randomly:
          - Patient A's Monday scan → Training
          - Patient A's Friday scan → Testing
          
        The model can "cheat" by recognizing Patient A's unique:
          - Rib cage shape
          - Heart position  
          - Spine curvature
          
        This leads to ARTIFICIALLY HIGH test accuracy that doesn't
        generalize to NEW patients in the real world.
        
        ═══════════════════════════════════════════════════════════════════
        THE SOLUTION:
        ═══════════════════════════════════════════════════════════════════
        
        Split by PATIENT ID, not by individual images:
        - ALL of Patient A's images → Training (or ALL → Testing)
        - Model has NEVER seen any image from test patients
        - Test accuracy reflects TRUE generalization
        
        Args:
            test_size (float): Fraction of patients for test set (default: 20%)
            val_size (float): Fraction of REMAINING training patients for 
                             validation (default: 12.5%, which is 10% of total)
                             
        Returns:
            tuple: (train_df, val_df, test_df) - Three DataFrames
        """
        if self.data is None:
            print("Data not loaded")
            return
            
        print('---------------------- Splitting dataset by PatientID --------------------------')
        
        # Step 1: Get list of UNIQUE patients
        patients = self.data['patientid'].unique()
        
        # Step 2: Split PATIENTS (not images) into train+val vs test
        train_p, test_p = train_test_split(patients, test_size=test_size, random_state=42)
        
        # Step 3: Split train PATIENTS into train vs validation
        train_p, val_p = train_test_split(train_p, test_size=val_size, random_state=42)
        
        # Step 4: Filter original DataFrame by patient membership
        train_df = self.data[self.data['patientid'].isin(train_p)]
        val_df = self.data[self.data['patientid'].isin(val_p)]
        test_df = self.data[self.data['patientid'].isin(test_p)]
        
        return train_df, val_df, test_df
