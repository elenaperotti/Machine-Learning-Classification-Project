## main.py ##

# It is the only python file that we need to run in order to obtain results.


import pandas as pd
import model
import preprocessor as prep

submissions = 2

def main():
    # 1: Load datasets
    train = pd.read_csv("development.csv")
    test = pd.read_csv("evaluation.csv")

    id = test["Id"]

    # 2: drop irrelevant features
    train = prep.feature_drop(train)
    test = prep.feature_drop(test)

    # 3: Split dataset in features X and target y
    y = train['death']  # -> death = target
    X = train.drop(columns=['death'])

    # 4: Select best model
    results_df, fitted_pipelines = model.select_best_model(X, y)

    best_results = results_df.head(submissions)
    print(f"Best models: ")
    for rank, row in enumerate(best_results.itertuples(index=False), start=1):
        print(f"  #{rank}: {row.model} | config: {row.config} | "
            f"f1_macro={row.best_cv_f1_macro:.4f} | params: {row.best_params}")


    # 5: Save a submission only for the 2 best models 
    for row in best_results.itertuples(index=False):
        key = (row.model, row.config)  
        pipeline = fitted_pipelines[key]
        preds = pipeline.predict(test)

        sub = pd.DataFrame({
            "Id": id,
            "Predicted": preds.astype(int),
        })
        
        filename = f"submission_{row.config}_{row.model}.csv"
        sub.to_csv(filename, index=False)
        print(f"File saved")
        


if __name__ == "__main__":
    main()

    


