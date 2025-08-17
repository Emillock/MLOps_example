import gzip
import os
import pickle
import pickletools
import warnings
from functools import partial

import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from xgboost import XGBClassifier

from utils import str_to_int_func, to_df_func, winsorize_array

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def save_model(filename: str, model: object):
    """
    Function saves model into pickle object.
    """
    file_path = os.path.join(ROOT_DIR, "models", filename)
    with gzip.open(file_path, "wb") as f:
        pickled = pickle.dumps(model)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)


def val_pattern():
    arr = []
    for second in range(1, 7):
        for first in range(2, 22):
            s = f"val{first}_{second}"
            if s in ("val3_6", "val3_5", "val3_4"):
                continue
            arr.append(s)
    return arr


def main():
    file_path = os.path.join(ROOT_DIR, "data", "processed", "multisim_dataset.parquet")
    df = pd.read_parquet(file_path)

    target = "target"
    numeric_cols = ["age", "tenure", "age_dev", "dev_num"]
    binary_cols = ["is_dualsim", "is_featurephone", "is_smartphone"]
    categorical_cols = [
        "trf",
        "gndr",
        "dev_man",
        "device_os_name",
        "simcard_type",
        "region",
    ]
    monthly_cols = val_pattern()

    features = numeric_cols + categorical_cols + binary_cols + monthly_cols

    df = df[features + [target]].copy()

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    str_to_int = FunctionTransformer(func=str_to_int_func, validate=False)

    winsor_transformer = FunctionTransformer(func=winsorize_array)

    to_df = FunctionTransformer(func=partial(to_df_func, cols=features), validate=False)

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("str_to_int", str_to_int),
                        ("impute", SimpleImputer(strategy="median")),
                        ("winsorize", winsor_transformer),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value="Missing"),
                        ),
                        ("encode", CatBoostEncoder()),
                    ]
                ),
                categorical_cols,
            ),
            (
                "bin",
                Pipeline(
                    [
                        ("str_to_int", str_to_int),
                        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
                    ]
                ),
                binary_cols,
            ),
            ("monthly", "passthrough", monthly_cols),
        ]
    )

    params = {
        "n_estimators": 186,
        "max_depth": 10,
        "learning_rate": 0.01155364929483116,
        "subsample": 0.5596769098694525,
        "colsample_bytree": 0.9364342412798315,
        "min_child_weight": 7,
    }

    model_pipeline = Pipeline(
        [("preproc", preprocessor), ("to_df", to_df), ("xgb", XGBClassifier(**params))]
    )

    # X_train_transformed = preproc_pipeline.fit_transform(X_train, y_train)

    # weights = preproc_pipeline.named_steps[weights_step_name].feature_weights_.tolist()

    model_pipeline.fit(X_train, y_train)

    # X_test_transformed = preproc_pipeline.transform(X_test)

    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy (before save): {acc:.3f}")
    print(classification_report(y_test, y_pred))

    filename = "xgb_trainmodel.pkl.gz"

    save_model(filename, model_pipeline)


if __name__ == "__main__":
    main()
