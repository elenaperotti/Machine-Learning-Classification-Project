# Predicting Survival of Critically Ill Patients

Binary classification pipeline for the DSL Lab project: predict the `death` outcome for critically ill patients, evaluated with
**macro F1-score**.


## 1. Files in this folder

main.py -> python file to run in order to obtain results: here .csv files are read, elaborated and the 2 best solutions are submitted in 
           another .csv file 

preprocessor.py - > python file that contains functions for the preprocessing part 

model.py' -> python file that contains functions for the model selection part

analysis.ipynb ->  Exploratory data analysis notebook: here is contained all the feature analysis with graphs represented, and the model 
                   selection is repeated in order to obtain graphs about the results too (ROC curve and Confusion Matrix)


## 2. Requirements

pandas
numpy
scikit-learn
matplotlib
seaborn       



## 3. Pipeline overview
This is the overview necessary to obtain the results

development.csv, evaluation.csv       -> read.csv() to read files
        │
        ▼
 preprocessor.feature_drop()          -> drops "Id", "temp_f" (redundant / non-informative)
        │
        ▼
 preprocessor.divide_features_full()  -> splits remaining columns into
 or divide_features_reduced()            numerical / ordinal / nominal 
        │
        ▼
 preprocessor.build_preprocessor()    -> ColumnTransformer for feature extraction and standardization 
        │
        ▼
 model.select_best_model_lr()         -> GridSearchCV over Logistic Regression,
 model.select_best_model_hgb()           GridSearchCV over HistGradientBoosting,
                                          one run per preprocessing config, 5-fold
                                          stratified CV, scoring = macro F1
        │
        ▼
 best (config, model) combination     -> refit on all of development.csv,
                                          predict on evaluation.csv,
                                          write submission.csv



#### PREPROCESSING

### preprocessing.py functions

1 - def feature_drop(df): 
-> function that eliminate useless feature from the dataset (in thid case, Id and temp_f)

2 - def divide_features_full(): 
-> function that divide features in numerical, categorical and ordinal

3 - def divide_features_reduced(df, red):
-> function that divide features in numerical, categorical and ordinal from a reduced dataset (different trials)

4 - def build_preprocessor(numerical_features,
        ordinal_features,
        categorical_features,
        ordinal_categories,
        hgb: bool = False,   
):
-> function that use Column Transformers to apply the feature extraction necessary for each model


#### MODEL

### model.py functions

1 - def get_lr_candidate():
-> function to call for the Hyperparameters tuning of Logistic Regression

2 - def get_hgb_candidate(categorical_features):
-> function to call for the Hyperparameters tuning of Hist Gradient Boosting Classifier

3 - def get_preprocessing_configs(X):
-> function that get all the features combinations to test 

4 - def _run_grid_search(pipe, param_grid, X, y, cv):
-> function that containg the Grid research to apply in the model selection

5 - def select_best_model_lr(X, y)
-> function to select the best model for Logistic Regression

6 - def select_best_model_hgb(X, y)
-> function to select the best model for Hist Gradient Boosting Classifier

7 - def select_best_model_hgb(X, y)
-> function that takes all the results obtained in the model selection and search for the best results between them.






