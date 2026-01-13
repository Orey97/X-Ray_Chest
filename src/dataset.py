import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Dataset:

    def __init__(self, data_path):
        self.file_path = data_path 
        self.data = None 
        self.load_data()

    def load_data(self):
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

    #modificare valori della colonna age da ---Y a --- (esempio 045Y a 45 o 105Y a 105)
    def modify_age_column(self, column_name):
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

    #serve per uniformare i nomi in lowercase/uppercase
    def clean_column_names(self):
        if self.data is None:
            print("Data not loaded")
            return
        print("---------------------- Cleaning column names --------------------------")
        
        self.data.columns = (
            self.data.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("(", "")
                .str.replace(")", "")
        )


    #pulizia delle eichette multi-label
    def clean_labels(self, column_name):
        if self.data is None:
            print("Data not loaded")
            return
        print(f'---------------------- Cleaning labels in column {column_name} --------------------------')
        self.data[column_name] = self.data[column_name].str.replace(" ","").str.split('|')

    #dobbiamo fare one-hot-encoding delle labels
    def one_hot_encode_labels(self, column="label"):
        unique_labels = sorted(
            set(label for sublist in self.data[column] for label in sublist)
        )
        for label in unique_labels:
            self.data[label] = self.data[column].apply(lambda x: int(label in x))

    #ai fini di CNN sono necessari solo image, patientid e le one hot encoded labels
    def select_relevant_columns(self):
        keep = ['image', 'patientid'] + [col for col in self.data.columns if col not in ['image', 'patientid', 'label']
        ]
        print(f"keepin columns: {keep}")
        self.data = self.data[keep]


    #dobbiamo fare un check se le immagini esistono nel path specificato sennò ci crash il training per file mancanti, corrotti...
    def check_image_files(self, image_dir):
        missing = [img for img in self.data['image'] if not os.path.exists(f"{image_dir}/{img}")]
        return missing
    
    #IMPORTANTE: rischio LEAKAGE se faccio split direttamente sul dataframe
    #stesso paziente può avere più immagini ed essendoci follow-up anche molto simili, rischio che il modello impari la persona e non la patologia
    #dobbiamo avere pazienti diversi nei set di train, val, test
    def patient_split(self, test_size=0.2, val_size=0.125): #attento val non può essere 10% di 80% 
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

    
    



    
    
    


