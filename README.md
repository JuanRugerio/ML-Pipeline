# AI Training Pipeline

## Environment building and Project execution
Docker environment provided. Simply run:  
docker compose build --no cache  
docker compose up

*Please format .csv input and new data files for each feature to lie at its own column, with the target varible under the name "target"*

## Project description

Engineering side: The Pipeline is properly modularized in components which are orchestrated by a file with such
purpose (run.sh). A Config file externalizes all modifying variables in a centralized location. The flow of proper
input components for each pipeline part is adecuately orchestrated by the run.sh file, and the outputs placed at
their corresponding subfiles. Representative naming is taken care of, comments reinforce the literary understanding
of the code, And perhaps at the most interesting, the code is organized in classes in an OOP fasion, inviting for
potential for growth as a more developed project.

Machine Learning side: The designed Pipeline begins with a Preprocessing step where missing data is imputed,
outliers caped, categorical variables ordinally encoded and numerical features scaled, it works with an XGBoost
model, due to its balance for simplicity and power, adecuate to this particular modeling scenario. An initial
parameter search is performed to gain early infromation about potential relevant features. Followed by a feature
selection procedure, based on SHAPLEY analysis. The relevance of the individual features in their capability of
explaining the outcome is reflected in the SHAP plots. Lastly, a fresh new XGBoost model is fit on the reduced
dataset with identified most promissing features, and finetuned with a Bayesian hyperparameter tuning strategy.
AUC Curves are observed through the lifecycle of the iterations, displaying promissing values. The final model can
produce predictions on datapoints for which independent, yet not the dependent variable are known.
