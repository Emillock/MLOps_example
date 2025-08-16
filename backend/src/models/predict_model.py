from xgboost import XGBClassifier
import pandas as pd
from typing import Iterable, Optional
import pickle
import os
import io
import gzip
import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def load_model(filepath: str):
    """
    Load a gzip-pickled model from disk.
    """
    with gzip.open(filepath, "rb") as f:
        p = pickle.Unpickler(f)
        clf = p.load()
    return clf


def _load_dataframe_from_bytes(
    file_content: bytes, filename: Optional[str]
) -> pd.DataFrame:
    """
    Read an uploaded file (bytes) into a DataFrame using file extension.
    Defaults to Excel if the extension is ambiguous.
    """
    name = (filename or "").lower()
    buffer = io.BytesIO(file_content)

    if name.endswith(".csv"):
        df = pd.read_csv(buffer)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(buffer)
    elif name.endswith(".parquet"):
        df = pd.read_parquet(buffer)
    else:
        # Fall back to Excel; FastAPI already validated extensions, this is just a guard.
        df = pd.read_excel(buffer)
    return df


def main(
    file_content: Optional[bytes] = None, filename: Optional[str] = None
) -> Iterable:

    if file_content:
        X_test = _load_dataframe_from_bytes(file_content, filename)
    else:
        # Fallback path for local testing
        df_path = os.path.join(ROOT_DIR, "data", "processed",
                              "multisim_dataset.parquet")
        df = pd.read_parquet(df_path)
        target = "target"

        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    model_path = os.path.join(ROOT_DIR, "models", "xgb_trainmodel.pkl.gz")

    xgb = load_model(model_path)

    y_pred = xgb.predict(X_test)

    try:
        return y_pred.tolist()
    except TypeError:
        return [p for p in y_pred]


if __name__ == "__main__":
    preds = main()
    print(f"Generated {len(preds)} predictions")
