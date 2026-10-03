# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Useless feature to drop from the dataset
drop = ['Id',"temp_f"]


### Numerical Features
num_feat = ["age", "scoma","edu", "charges", "totcst", "totmcst",
    "avtisst", "sps", "aps", "surv2m", "surv6m","prg2m", "prg6m", "hday", "dnrday",
    "meanbp", "wblc", "resp", "temp", "hrt","pafi","alb","crea","ph", "glucose", "bun",
    "adls","adlsc", "adlp", "bili","urine","sod","diabetes", "dementia"
]

### Ordinal Features
ord_feat = ["ca","income","num.co"]
ord_categories = {
    "income": ["under $11k", "$11-$25k","$25-$50k", ">$50k"],
    "ca": ["no", "yes", "metastatic"],
    "num.co": [0,1,2,3,4,5,6,7,8,9]
}

### Categorical Features
cat_feat = [
    "sex", "dzgroup","dzclass", "race", "dnr", "medical_center",
    "admission_month",
]

# 1: Drop Feature that we considered irrelevant in analysis.ipynb
def feature_drop(df):
    df = df.drop(columns = drop)
    return df

# 2: function to divide feature in numerical, ordinal and categorical
def divide_features_full():
    ordinal_categories = [ord_categories[f] for f in ord_feat]
    return num_feat, ord_feat, cat_feat, ordinal_categories


# 2: Same thing for a reduced number of features -> drop features that may be useless (explained in analysis.ipynb) 
# -> different trials in model.py
def divide_features_reduced(df, red):
    df = df.drop(columns = red)
    numerical = [col for col in num_feat if col in df.columns]
    ordinal = [col for col in ord_feat if col in df.columns]
    categorical = [col for col in cat_feat if col in df.columns]

    ordinal_categories = [ord_categories[f] for f in ordinal]

    return numerical, ordinal, categorical, ordinal_categories


# 3: function that handle missing values and preprocess features
def build_preprocessor(
        numerical_features,
        ordinal_features,
        categorical_features,
        ordinal_categories,
        hgb: bool = False,   
):
    if hgb:
        # HistGradientBoostingClassifier
        preprocessor = ColumnTransformer(
            transformers=[
                ("ord", OrdinalEncoder(categories=ordinal_categories,
                                        handle_unknown="use_encoded_value",
                                        unknown_value=-1), ordinal_features),
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                        unknown_value=-1,
                                        encoded_missing_value=-1), categorical_features),
                ("num", "passthrough", numerical_features),   
            ],
            remainder="drop",  
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
    else:
        # Logistic Regression 
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy='median')), 
                    ("scaler", StandardScaler()), 
                ]), numerical_features),
                ("ord", Pipeline([
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("scaler", OrdinalEncoder(categories=ordinal_categories,
                                              handle_unknown="use_encoded_value",
                                              unknown_value=-1)),
                ]), ordinal_features),
                ("cat", Pipeline([
                        ("imputer", SimpleImputer(strategy='most_frequent')),
                        ("scaler", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]), categorical_features)
            ], remainder='drop'
        )

    steps = [("preprocessor", preprocessor)]

    return Pipeline(steps=steps)