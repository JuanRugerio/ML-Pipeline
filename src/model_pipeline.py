# Stori is cool
"""
Model training pipeline for ML project (OOP).

This script:
- Loads processed data
- Performs cross-validation
- Runs initial hyperparameter search
- Performs final Bayesian hyperparameter optimization
- Trains the final model
- Saves model artifacts

Usage:
    python src/model_pipeline.py --config config.yaml --input outputs/processed_data.csv --output outputs/
"""
import sys
import argparse
import os
import joblib
import yaml
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from visualization import Visualizer

#(33)
#En línea recibiendo config, input, output, espacio para el modelo, para X, y
#cv con n_splits y random state desde config.
#load_dataset: lee los datos procesados para train, separa en X y y. 
#Guarda en X_test y y_test. 
#initial_hyperparam_search: Toma param_grid y n_iter del config, igual xgbdeafults
#arma el xbgclassifier con los hyperparameters, inicia un randomized search con 
#xgb, param grip, n iter, roc auc, cv, hace el fit con los datos de entrenamiento
#guarda en el espacio del objeto de modelo es que tuvo los mejores hps, e imprime
#los hps de él
#final_bayesian_search: Toma grid y n_trials de config, toma xbg deafault como 
#parámetros y agrega el random state de training a xgb defaults, guarda en cv
#preds para las visualizaciones, en objective para cada trial define y toma aquí
# el espacio para los diferentes hps desde el config, arma el xgbclassifier 
#con los params, divide los datos en X_train, X_test, y_train, y_test, según 
#cv, hace el fit del modelo con los datos de entrenamiento y luego hace 
#predicciones para la parte test, guarda en cv scores los auc y en los índices
#de los datapoints que fueron para test las predicciones, todo eso es objective 
#y lo llama n iter veces, guarda los best_params y los imprime y guarda el modelo
#con la mejor configuración. 
#train_final_model: Toma el tamanio del test del config, separa los datos entre
#train y test. Hace el fit del model con los datos de entrenamiento, testea
#y guarda en evals_result
#save_model: Toma el path del config, y guarda el modelo como final_model.pkl
#run: llama load_data, initial, final y cross validate
#load_config: Mismo
#main: Carga el config, instancia el objeto, llama a run?


class Trainer:
    """Trainer class for XGBoost classifier."""

    def __init__(self, config: dict, input_path: str, output_dir: str):
        self.config = config
        self.input_path = input_path
        self.output_dir = output_dir
        self.model = None
        self.X = None
        self.y = None
        self._trial_results = []
        self.X_test = None
        self.y_test = None
        self.cv = StratifiedKFold(n_splits=self.config["training"]["cv_folds"],
                                  shuffle=True,
                                  random_state=self.config["training"].get("random_state", 42)) #If not such parameter found, use 42

    def load_data(self):
        """Load processed dataset and split features/target. Load train/test separately."""
        df_train = pd.read_csv(self.input_path)
        self.X = df_train.drop(columns=[self.config["data"]["target_column"]])
        self.y = df_train[self.config["data"]["target_column"]]

        # Load test set from the same folder
        test_path = self.config["paths"].get("test_data_path")  # Add path in config
        if test_path and os.path.exists(test_path):
            df_test = pd.read_csv(test_path)
            self.X_test = df_test.drop(columns=[self.config["data"]["target_column"]])
            self.y_test = df_test[self.config["data"]["target_column"]]
        else:
            self.X_test = None
            self.y_test = None

        return self


    def initial_hyperparam_search(self):
        """Randomized search over initial hyperparameters."""
        param_grid = self.config["tuning"]["initial_search"]["param_grid"]
        n_iter = self.config["tuning"]["initial_search"]["n_iter"]

        xgb_defaults = self.config["tuning"].get("xgboost_defaults")
        xgb = XGBClassifier(**xgb_defaults, random_state=self.config["training"].get("random_state", 42)) 
        search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=self.cv,
            verbose=1, #How much info about processed is shown, here, basic progress
            n_jobs=-1, #Controls number of CPU cores used, here all available CPU cores
            random_state=self.config["training"].get("random_state", 42) 
        )
        search.fit(self.X, self.y)
        self.model = search.best_estimator_
        print(f"✅ Initial hyperparameter search complete. Best params: {search.best_params_}")
        return self

    def final_bayesian_search(self, n_trials=None):
        """
        Perform final Bayesian hyperparameter optimization using Optuna.
        Updates self.model with the best found XGBClassifier, but preserves
        CV folds for later visualization if needed.
        """

        final_config = self.config["tuning"]["final_search"]
        param_bounds = final_config["param_bounds"]
        n_trials = final_config.get("n_iter", 30) if n_trials is None else n_trials

        xgb_defaults = self.config["tuning"].get("xgboost_defaults", {})
        random_state = self.config["training"].get("random_state", 42)
        xgb_defaults["random_state"] = random_state  # enforce reproducibility

        # Store CV predictions (optional, for visualization later)
        self._cv_preds = []

        def objective(trial): 
            params = {
                "n_estimators": trial.suggest_int("n_estimators", *param_bounds["n_estimators"]),
                "max_depth": trial.suggest_int("max_depth", *param_bounds["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *param_bounds["learning_rate"], log=True),
                "subsample": trial.suggest_float("subsample", *param_bounds["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *param_bounds["colsample_bytree"]),
                "reg_alpha": trial.suggest_float("reg_alpha", *param_bounds["reg_alpha"]),
                "reg_lambda": trial.suggest_float("reg_lambda", *param_bounds["reg_lambda"]),
                **xgb_defaults
            }

            model = XGBClassifier(**params)
            cv_scores = []

            # Save fold predictions for visualization
            fold_preds = np.zeros(len(self.y))

            for train_idx, val_idx in self.cv.split(self.X, self.y):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
                model.fit(X_train, y_train)
                preds = model.predict_proba(X_val)[:, 1]
                cv_scores.append(roc_auc_score(y_val, preds))
                fold_preds[val_idx] = preds

            self._trial_results.append((trial.number, fold_preds.copy(), self.y.copy()))
            self._cv_preds.append(fold_preds)
            return np.mean(cv_scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        best_params = study.best_params
        print(f"✅ Final Bayesian search complete. Best params: {best_params}")

        merged_params={**xgb_defaults, **best_params}
        merged_params["random_state"] = random_state
        merged_params.setdefault("eval_metric", "logloss")
        self.model = XGBClassifier(**merged_params)

        joblib.dump(self._trial_results, os.path.join(self.output_dir, "trial_results.pkl"))

        # NOTE: Do NOT fit on full dataset here if you need fold predictions for visualization
        # Instead, let train_final_model() handle full dataset fitting
        return self


    def train_final_model(self, return_eval_results=False):
        """Fit model on full dataset, optionally with a validation split for evals_result."""
        # Stratified train/validation split
        test_size = self.config["training"].get("validation_split", 0.1)
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_test,
            self.y_test,
            test_size=test_size,
            stratify=self.y_test,
            random_state=self.config["training"].get("random_state", 42)
        )


        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        y_pred_proba = self.model.predict_proba(X_val)[:,1]
        os.makedirs(self.output_dir, exist_ok=True)
        joblib.dump((y_val, y_pred_proba), os.path.join(self.output_dir, "final_model_preds.pkl"))



        return self



    def save_model(self):
        """Save trained model to model_dir specified in config."""
        model_dir = self.config["paths"]["model_dir"]
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "final_model.pkl")
        joblib.dump(self.model, model_path)
        print(f"✅ Model saved to {model_path}")
        return model_path

    def run(self):
        """Run full training pipeline."""
        self.load_data() \
            .initial_hyperparam_search() \
            .save_model()
        sys.exit(0)

    def run_final(self):
        """Run second stage training pipeline (after feature selection)."""
        self.load_data() \
            .final_bayesian_search()\
        # Capture evals_result from training with validation split
        self.train_final_model()

        vis = Visualizer(self.config, self.output_dir)

        trial_path=os.path.join(self.output_dir, "trial_results.pkl")
        if os.path.exists(trial_path):
            trial_results = joblib.load(trial_path)
            for trial_num, y_pred_proba, y_true in trial_results:
                vis.plot_roc_curve(y_true, y_pred_proba, label=f"trial_{trial_num}")

        final_preds_path = os.path.join(self.output_dir, "final_model_preds.pkl")
        if os.path.exists(final_preds_path):
            y_true, y_pred_proba = joblib.load(final_preds_path)
            vis.plot_roc_curve(y_true, y_pred_proba, label="final_model")
        self.save_model()
        
        # Return evals_result so external code can use it for visualization
        return self

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        return config


if __name__ == "__main__":
    import yaml
    import argparse
    from visualization import Visualizer
    from model_pipeline import Trainer, load_config  # ensure Trainer and load_config are importable
    import shap
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import sys

    parser = argparse.ArgumentParser(description="Train ML model.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--input", type=str, required=True, help="Path to preprocessed CSV file")
    parser.add_argument("--output", type=str, required=True, help="Directory for output files")
    parser.add_argument("--stage", type=int, choices=[0,1], required=True,
                        help="0 = initial run (pre-feature selection, 1 = final run (post-feature selection) )")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize trainer
    trainer = Trainer(config, args.input, args.output)

    # Run training pipeline and get evals_result for plotting

    if args.stage == 0:
        trainer.run()  # run() now returns evals_result
    elif args.stage == 1:
        trainer.run_final()
    else:
        sys.exit("Invalid stage. Use 0 or 1")
