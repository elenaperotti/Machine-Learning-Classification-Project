# Model
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

import preprocessor as prep


random = 42
CV = 5
score = "f1_macro"


def get_lr_candidate():
    return (
        LogisticRegression(max_iter=2000, random_state=random),
        {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__class_weight": [None,"balanced"],
        },
    )

def get_hgb_candidate(categorical_features):

    return (
        HistGradientBoostingClassifier(
            random_state= random,
            categorical_features=categorical_features,
        ),
        {
            "classifier__max_iter": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [None, 6, 10],
            "classifier__class_weight": [None,"balanced"],
        },
    )


def get_preprocessing_configs(X):
    numerical, ordinal, nominal, categorical = prep.divide_features_full()

    set = {
        "reduced_edu": ["edu"],
        "reduced_adls+adlp": ["adlp", "adls"],
        "reduced_hrt_edu": ["hrt","edu"],
        "reduced_totcst": ["totcst"],
        "reduced_class": ["dzclass"],
        "reduced_all": ["edu", "hrt", "totcst","dzclass","adls","adlp"],
    }

    configs = {
        "full": dict(numerical=numerical, ordinal=ordinal, nominal=nominal,
                      categories=categorical),
    }

    for name, drop_list in set.items():
        red_num, red_ord, red_nom, red_cat = prep.divide_features_reduced(X, drop_list)
        configs[name] = dict(numerical=red_num, ordinal=red_ord, nominal=red_nom,
                              categories=red_cat)

    return configs


def _run_grid_search(pipe, param_grid, X, y, cv):
    grid = GridSearchCV(
        pipe, param_grid=param_grid, scoring= "f1_macro",
        cv=cv, n_jobs=-1, refit=True,
    )
    grid.fit(X, y)
    return grid


def select_best_model_lr(X, y):
    configs = get_preprocessing_configs(X)
    cv = StratifiedKFold(n_splits=CV, shuffle=True, random_state=random) # -> keep the same proportion of target in the different folds
                                                                         # useful in this case because target is not balanced

    rows = []
    fitted_pipelines = {}

    for config_name, cfg in configs.items():
        estimator, param_grid = get_lr_candidate()
        pre_pipeline = prep.build_preprocessor(
            cfg["numerical"], cfg["ordinal"], cfg["nominal"], cfg["categories"], hgb=False,
        )
        pipe = Pipeline(steps=list(pre_pipeline.steps) + [("classifier", estimator)])

        grid = _run_grid_search(pipe, param_grid, X, y, cv)
        fitted_pipelines[config_name] = grid.best_estimator_

        rows.append({
            "config": config_name,
            "model": "LogisticRegression",
            "best_cv_f1_macro": grid.best_score_,
            "best_params": grid.best_params_,
        })
        print(f"[{config_name:14s} | LogisticRegression  ] CV f1_macro = {grid.best_score_:.4f}")

    results_df = pd.DataFrame(rows).sort_values("best_cv_f1_macro", ascending=False).reset_index(drop=True)
    return results_df, fitted_pipelines


def select_best_model_hgb(X, y):
    configs = get_preprocessing_configs(X)
    cv = StratifiedKFold(n_splits=CV, shuffle=True, random_state=random)

    rows = []
    fitted_pipelines = {}

    for config_name, cfg in configs.items():
        estimator, param_grid = get_hgb_candidate(cfg["nominal"])
        pre_pipeline = prep.build_preprocessor(
            cfg["numerical"], cfg["ordinal"], cfg["nominal"], cfg["categories"], hgb=True,
        )
        pipe = Pipeline(steps=list(pre_pipeline.steps) + [("classifier", estimator)])

        grid = _run_grid_search(pipe, param_grid, X, y, cv)
        fitted_pipelines[config_name] = grid.best_estimator_

        rows.append({
            "config": config_name,
            "model": "HistGradientBoosting",
            "best_cv_f1_macro": grid.best_score_,
            "best_params": grid.best_params_,
        })
        print(f"[{config_name:14s} | HistGradientBoosting] CV f1_macro = {grid.best_score_:.4f} ")

    results_df = pd.DataFrame(rows).sort_values("best_cv_f1_macro", ascending=False).reset_index(drop=True)
    return results_df, fitted_pipelines


def select_best_model(X, y):

    lr_results, lr_pipelines = select_best_model_lr(X, y)
    hgb_results, hgb_pipelines = select_best_model_hgb(X, y)

    results_df = pd.concat([hgb_results, lr_results], ignore_index=True)
    results_df = results_df.sort_values("best_cv_f1_macro", ascending=False).reset_index(drop=True)

    fitted_pipelines = {("LogisticRegression", k): v for k, v in lr_pipelines.items()}
    fitted_pipelines.update({("HistGradientBoosting", k): v for k, v in hgb_pipelines.items()})

    return results_df, fitted_pipelines