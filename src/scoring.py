# Stori is cool
"""
Module for loading a trained model and generating predictions on new data.
"""

import os
import joblib
import pandas as pd
import numpy as np
import tempfile
#(18)
#Guarda config, tiene espacio para modelo, scaler, encoder, arma direcciones a 
#final_model.pkl, scaler.pkl y enconder.pkl
#load_model_and_preprocessors: Si el modelo está lo carga a model, 
#lo mismo con scaler y encoder
#preprocess_new_data: Recibe un df y hace una copia a df_processed, 
#si hay encoder, es ordinal, selecciona las columnas categoricas y aplica,
#si no, es por one hot, toma los features del modelo, aplica,
#si hay alguna variable en el modelo que no esté en el resultado las mete 
#con valores 0, avisa si hay features en el resultantes que no estuvieran en 
#features y hace una subselección a sólo lo que si estaba.
#Pone en num_cols, sólo las numéricas, las escala
#predict: Recibe input_path, si no hay modelo lanza error, 
#lee datos del que sea el input_path en config aquí, si el target está en 
#el archivo lo desecha, llama preprocesar y pasa los datos sin target, 
#hace predicciones de probabilidades, saca el output dir del config,
#guarda las probabilidades y las predicciones en el path. 
#main: Parser, config, instancia y pasa config, llama load model and preprocessors, 
#llama predict y pasa args, output filename y output dir

class Scorer:
    def __init__(self, config):
        self.config = config
        self.model = None

        model_dir = self.config["paths"]["model_dir"]
        self.model_path = os.path.join(model_dir, "final_model.pkl")

    def load_model_and_preprocessors(self):
        """Load trained model and preprocessing artifacts."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Model not found at {self.model_path}")
        self.model = joblib.load(self.model_path)

    def preprocess_new_data(self, input_path: str) -> pd.DataFrame:
        """
        Preprocess new data using the exact same pipeline from preprocess.py.
        """
        from preprocess import Preprocessor  # import here to avoid circular import

        output_dir = self.config["paths"].get("predictions_dir", "outputs/predictions")
        os.makedirs(output_dir, exist_ok=True)
        preprocessor = Preprocessor(self.config, input_path, output_dir=output_dir)
        processed_df = preprocessor.transform_and_return()

        # Drop target column if still present
        target_col = self.config["data"].get("target_column", "target")
        if target_col in processed_df.columns:
            processed_df = processed_df.drop(columns=[target_col])

        return processed_df


    def predict(self, input_path, output_filename="predictions.csv", output_dir=None):
        """
        Generate predictions for new data (without target), applying saved scaler/encoder.
        """
        if self.model is None:
            raise ValueError("❌ Model not loaded. Call load_model_and_preprocessors() first.")

        # Preprocess new data
        new_data_processed = self.preprocess_new_data(input_path)

        # Predict probabilities
        preds = self.model.predict_proba(new_data_processed)[:, 1]

        # Build output path
        if output_dir is None:
            output_dir = self.config["paths"]["predictions_dir"]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        # Save results
        results = pd.DataFrame({
            "probability": preds,
            "prediction": (preds >= 0.5).astype(int)
        })
        results.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to {output_path}")

        return results


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Generate predictions with trained model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Path to new data CSV")
    parser.add_argument("--output", type=str, required=False, help="Directory for output files")
    parser.add_argument("--filename", type=str, default="predictions.csv", help="Output file name")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    scorer = Scorer(config)
    scorer.load_model_and_preprocessors()
    scorer.predict(
        args.input,
        output_filename=args.filename,
        output_dir=args.output
    )



