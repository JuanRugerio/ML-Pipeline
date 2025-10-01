# Stori is cool
"""
Feature selection module for ML pipeline (OOP).

Performs iterative SHAP-based elimination:
- Compute SHAP values for current feature set
- Rank features by mean absolute SHAP value
- Remove lowest importance features until all features go over importante threshold, or reaching min_features

Usage:
    python src/feature_selection.py --config config.yaml --input outputs/processed_data.csv --output outputs/
"""

import argparse
import os
import yaml
import pandas as pd
import numpy as np
import joblib
import shap
from xgboost import XGBClassifier
#(20)
#Instancia y guarda config, input path output dir, en el momento de ser llamado
#el modelo que recibe es el resultante del primer tuning, espacio para X, y y selected_features.
#load_data: Aguas con processed_data! Parte en X y y
#iterative_shap_elimination: Toma min_features, elimination_step e importance_threshold
#del config, guarda todas las columnas de X en features, mientas haya más features
#en features que el mínimo requerido, hace un fit del modelo con un subset de features en features
#saca SHAP values, luego promedio por feature, hace lista ordenada de features y sus SHAP
#por SHAP, se queda con lo que tiene menor SHAP que el requerido, si no hay, 
#sale del retirador de variables, si hay meter en lista los primeros 
# (lo que esté definido en config), y reduce features a él mismo sin esas
#imprime cuáles se fueron y quedaron, al final imprime las que se quedaron 
#al final de todo.
#save_selected_features: Hace un csv con los nombres de los features seleccionados
#save_reduced_dataset: Lee de input, ojo otra vez con processed_data, 
#arma un subset del dataset con los elegidos e y, guarda en el output como
#processed_selected
#def run: Corre load data, iterative shap elimination, save selected features y save reduced dataset
#Carga el Config
#main con Parser de parametros, carga el config y el modelo, instancia objeto 
#y corre el run()

class FeatureSelector:
    """Performs iterative SHAP-based feature elimination."""

    def __init__(self, config: dict, input_path: str, output_dir: str, model=None):
        self.config = config
        self.input_path = input_path
        self.output_dir = output_dir
        self.model = model  # initial model from initial HP tuning
        self.X = None
        self.y = None
        self.selected_features = None

    def load_data(self):
        """Load processed dataset."""
        df = pd.read_csv(self.input_path)
        target_col = self.config["data"]["target_column"]
        self.X = df.drop(columns=[target_col])
        self.y = df[target_col]
        return self
 
    def iterative_shap_elimination(self):
        """
        Iteratively remove low-importance features based on SHAP values.
        Stops when no features fall below threshold or min_features is reached.
        """
        min_features = self.config["feature_selection"]["min_features"]
        elimination_step = self.config["feature_selection"]["elimination_step"]
        importance_threshold = self.config["feature_selection"].get("importance_threshold", 0.01) 

        features = self.X.columns.tolist()

        while len(features) > min_features:
            # Fit model on current features
            self.model.fit(self.X[features], self.y)

            # Compute SHAP values
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer(self.X[features]).values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = pd.Series(mean_abs_shap, index=features)
            feature_importance = feature_importance.sort_values(ascending=True)

            # Identify features below threshold
            low_importance = feature_importance[feature_importance < importance_threshold]

            if low_importance.empty:
                print("✅ No more low-importance features. Stopping elimination.")
                break

            # Remove up to elimination_step lowest ones
            to_remove = low_importance.index[:elimination_step].tolist()
            features = [f for f in features if f not in to_remove]

            print(f"Iteration: remaining features = {len(features)}")
            print(f"Removed features: {to_remove}")

        self.selected_features = features
        print(f"✅ Final selected features ({len(features)}): {features}")
        return self


    def save_selected_features(self):
        """Save list of selected features for downstream use."""
        os.makedirs(self.output_dir, exist_ok=True)
        selected_path = os.path.join(self.output_dir, "selected_features.csv")
        pd.DataFrame(self.selected_features, columns=["feature"]).to_csv(selected_path, index=False)
        print(f"✅ Selected features saved to {selected_path}")
        return self
    
    def save_reduced_dataset(self):
        """Save reduced dataset with selected features + target."""
        df = pd.read_csv(self.input_path)
        reduced = df[self.selected_features + [self.config["data"]["target_column"]]]
        reduced_path = os.path.join(self.output_dir, "processed_selected.csv")
        reduced.to_csv(reduced_path, index=False)
        print(f"✅ Reduced dataset saved to {reduced_path}")
        return self
    
    def run(self):
        """Run full feature selection pipeline."""
        (
            self.load_data()
            .iterative_shap_elimination()
            .save_selected_features()
            .save_reduced_dataset() 
        )

        from visualization import Visualizer
        visualizer = Visualizer(self.config, self.output_dir)
        visualizer.plot_shap_summary(self.model, self.X[self.selected_features], stage="post_selection")

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature selection via SHAP.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--input", type=str, required=True, help="Path to preprocessed CSV file")
    parser.add_argument("--output", type=str, required=True, help="Directory for output files")
    parser.add_argument("--model", type=str, required=True, help="Path to initial model (from initial HP tuning)")
    args = parser.parse_args()

    config = load_config(args.config)
    # Load initial model
    model = joblib.load(args.model)
    selector = FeatureSelector(config, args.input, args.output, model=model)
    selector.run()
