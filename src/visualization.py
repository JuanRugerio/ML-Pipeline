# Stori is cool
"""
Visualization utilities for model evaluation and interpretability.
"""

import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import shap
import numpy as np
(13)
#Guarda el config, se fija en el config el output dir como plots dir y lo guarda, 
#plot_roc: Recibe modelo, X test, y test? Hace predicciones con el modelo y 
#X test, con y test y y pred proba saca fpr y tpr, y el roc auc. 
#Hace el plot con eso. Guarda como roc_curve.png. 
#plot_loss_curves: Recibe evals_result, extrae val_metric de evals result, 
#plotea train vs test, guarda en output con loss_curves.png. 
#plot_shap_summary: Recibe modelo, X, aplica SHAP a ambos, guarda.
#plot_shap_dependence: Recibe modelo, X, feature, aplica SHAP a modelo y X,
#arma un dependence plot para 1 feature y lo guarda.
#main: config, args, toma el modelo del que está guardado, toma processed_data_test
#lo lee a df_test, X se queda con todo menos y e y con y. Instancia Visualizer
#corre plot_roc con el modelo X y y, corre plot shap summary con el modelo y X
#también a plot shap dependence si existe X.columns

class Visualizer:
    def __init__(self, config: dict, output_dir:str):
        """
        Initialize Visualizer.

        Parameters
        ----------
        config : dict
            Configuration dictionary with visualization settings.
        output_dir : str
            Directory where plots will be saved.
        """
        self.config = config
        self.output_dir = output_dir
        self.plots_dir = config["paths"]["plots_dir"]
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_roc_curve(self, y_true, y_pred_proba, label="final_model"):
        """Plot ROC curve and save to file."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"{label} (AUC = {roc_auc:.2f})") #lw linewidth
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        path = os.path.join(self.plots_dir, f"roc_curve_{label}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"✅ ROC curve saved to {path}")

    def plot_shap_summary(self, model, X, stage="post_selection"):
        """Generate SHAP summary plot."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X).values

        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        path = os.path.join(self.plots_dir, f"shap_summary_{stage}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"✅ SHAP summary plot saved to {path}")

if __name__ == "__main__":
    import argparse
    import joblib
    import yaml
    import pandas as pd
        
    parser = argparse.ArgumentParser(description="Generate evaluation plots.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--output", type=str, required=True, help="Directory for output files")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)


