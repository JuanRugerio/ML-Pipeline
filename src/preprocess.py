# Stori is cool
"""
Preprocessing module for ML pipeline (OOP version).

This script:
- Reads input CSV
- Cleans missing values
- Handles outliers
- Encodes categorical variables
- Scales numeric variables
- Outputs processed dataset

Usage:
    python src/preprocess.py --config config.yaml --input data/input.csv --output outputs/
"""


import argparse
import yaml
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#(25)
#En línea recibiendo config, input, output, toma de config nombre de variable 
#target, tiene un espacio para scaler y encoder, tiene un espacio para el df
#para X y para y, 
# #load_data lee el único archivo en input_path, en X 
#guarda todo menos target_col, en y guarda target_col.
#handle_missing: Toma estretegias del Config, itera por todas las columnas 
#predictivas, depende de si es numérica o categórica y el caso en config 
#aplica operación
#cap_outliers: Itera sobre columnas numéricas, para cada una calcula el valor del 
#cuantil 1 y 99 y todo valor que pase para arriba o abajo de ellos lo convierte en ellos.
#encode_categoricals: Checa primero si es one hot o cardinal en el config, 
#mete columnas categoricas en cat_cols, si es one hot la estrategia, convierte 
#las categorias en sus versiones one hot, si es ordinal asigna un numero a cada 
#categoria y transforma así los valores de las variables, si en test data hay categorías que no 
#estaban en training, pone un valor default. 
#Scale_numeric: Si desactivado en config, regresa como está, si si, filtra columnas
#numéricas y transforma
#save: Concatena la nueva version de X y y en processed_df, toma el df preprocesado
#y la variable de proporción del config y hace la división del dataset en train y test
#los escribe en el output como processed_dateset_train y processed_dataset_test
#en model_dir, guarda scaler y encoder
#run() corre todo
#Una funcion para abrir el config 
#El main pasa los argumentos y llama funcion para cargar config y luego instancia
#y corre el run

class Preprocessor:
    """Preprocesses tabular data according to config settings."""

    def __init__(self, config: dict, input_path: str, output_dir: str):
        self.config = config
        self.input_path = input_path
        self.output_dir = output_dir
        self.target_col = config["data"]["target_column"]
        self.scaler = None
        self.encoder = None

        # data holders
        self.df = None
        self.X = None
        self.y = None

    def load_data(self):
        """Read CSV and split into features + target."""
        self.df = pd.read_csv(self.input_path)
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found. Found columns: {self.df.columns.tolist()}")
        self.X = self.df.drop(columns=[self.target_col])
        self.y = self.df[self.target_col]
        self.X.to_csv(os.path.join(self.output_dir, "debug_step1.csv"), index=False)
        return self
    


    def handle_missing(self):
        """Handle missing values (numeric + categorical separately)."""
        num_strategy = self.config["preprocessing"]["handle_missing_numeric"]
        cat_strategy = self.config["preprocessing"]["handle_missing_categorical"]

        for col in self.X.columns:
            if pd.api.types.is_numeric_dtype(self.X[col]):
                if num_strategy == "median":
                    self.X[col].fillna(self.X[col].median(), inplace=True)
                elif num_strategy == "mean":
                    self.X[col].fillna(self.X[col].mean(), inplace=True)
            else:  # categorical
                if cat_strategy == "mode":
                    self.X[col].fillna(self.X[col].mode()[0], inplace=True) #[0] since mode() returns a series of scalars tied as a mode, we work with the first 
                elif cat_strategy == "constant":
                    self.X[col].fillna("missing", inplace=True)
        self.X.to_csv(os.path.join(self.output_dir, "debug_step2.csv"), index=False)
        return self

    def cap_outliers(self):
        """Cap numeric outliers at 1st and 99th percentiles."""
        if self.config["preprocessing"]["outlier_strategy"] != "cap":
            return self

        for col in self.X.select_dtypes(include=[np.number]).columns:
            lower, upper = self.X[col].quantile(0.01), self.X[col].quantile(0.99)
            self.X[col] = np.clip(self.X[col], lower, upper)
        self.X.to_csv(os.path.join(self.output_dir, "debug_step3.csv"), index=False)
        return self

    def encode_categoricals(self):
        """Encode categorical features as 0/1 integers without changing variable names."""
        encoding = self.config["preprocessing"]["categorical_encoding"]
        cat_cols = self.X.select_dtypes(exclude=[np.number]).columns.tolist()

        if not cat_cols:
            return self

        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self.X[cat_cols] = self.encoder.fit_transform(self.X[cat_cols]).astype(int)

        self.X.to_csv(os.path.join(self.output_dir, "debug_step4.csv"), index=False)
        return self


    def scale_numeric(self):
        """Scale numeric features if enabled in config."""
        if not self.config["preprocessing"].get("scale_numeric", False):
            return self

        self.scaler = StandardScaler()
        self.X = self.X.apply(pd.to_numeric, errors="ignore")
        num_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        self.X[num_cols] = self.scaler.fit_transform(self.X[num_cols])
        self.X.to_csv(os.path.join(self.output_dir, "debug_step5.csv"), index=False)
        return self

    def save(self):
        """Save processed dataset + preprocessing objects, and split into train/test."""
        processed_df = pd.concat([self.X, self.y], axis=1)
        os.makedirs(self.output_dir, exist_ok=True)

        # Stratified split
        test_size = self.config["preprocessing"].get("test_split", 0.2)
        train_df, test_df = train_test_split(
            processed_df,
            test_size=test_size,
            stratify=self.y, #Makes sure the split in y has the same balance for train and test size as in original data
            random_state=self.config["training"].get("random_state", 42)
        )

        train_path = os.path.join(self.output_dir, "processed_data_train.csv")
        test_path = os.path.join(self.output_dir, "processed_data_test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"✅ Preprocessing complete. Train saved to {train_path}, Test saved to {test_path}")
        return train_path, test_path

    def run(self):
        """Run full preprocessing pipeline in order."""
        (
            self.load_data()
            .handle_missing()
            .cap_outliers()
            .encode_categoricals()
            .scale_numeric()
            .save()
        )

    def transform_and_return(self) -> pd.DataFrame:
        """
        Run preprocessing pipeline and return processed dataset
        (without saving or splitting).            """
        (
            self.load_data()
            .handle_missing()
            .cap_outliers()
            .encode_categoricals()
            .scale_numeric()
        )
        processed_df = pd.concat([self.X, self.y], axis=1)
        return processed_df



def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess input data.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Directory for output files")
    args = parser.parse_args()

    config = load_config(args.config)
    preprocessor = Preprocessor(config, args.input, args.output)
    preprocessor.run()

