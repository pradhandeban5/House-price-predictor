"""Feature-aware preprocessing, model training, evaluation, and persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "pipeline.joblib"
MODEL_FORMAT_VERSION = 1


def _categorical_strings(values: np.ndarray) -> np.ndarray:
    """Give mixed Excel object columns one consistent representation."""
    values = np.asarray(values, dtype=object)
    missing = pd.isna(values)
    normalized = values.astype(object, copy=True)
    normalized[~missing] = values[~missing].astype(str)
    normalized[missing] = np.nan
    return normalized


def build_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    n_estimators: int = 350,
    max_depth: int | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Build a preprocessing and random-forest regression pipeline."""
    transformers: list[tuple[str, Pipeline, Sequence[str]]] = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, list(numeric_features)))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "stringify",
                    FunctionTransformer(
                        _categorical_strings, feature_names_out="one-to-one"
                    ),
                ),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, list(categorical_features))
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_cache_key(
    frame: pd.DataFrame,
    features: Sequence[str],
    n_estimators: int,
    max_depth: int | None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> str:
    """Fingerprint training data and settings so stale models are not reused."""
    try:
        row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    except TypeError:
        row_hashes = pd.util.hash_pandas_object(
            frame.astype("string").fillna(""), index=True
        ).to_numpy()
    dataset_hash = hashlib.sha256(row_hashes.tobytes()).hexdigest()
    descriptor = {
        "dataset": dataset_hash,
        "features": list(features),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "test_size": test_size,
        "random_state": random_state,
        "sklearn": sklearn.__version__,
        "format": MODEL_FORMAT_VERSION,
    }
    return hashlib.sha256(
        json.dumps(descriptor, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _feature_importance(
    pipeline: Pipeline,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    """Sum encoded-level forest importance back to the selected input columns."""
    forest = pipeline.named_steps["regressor"]
    transformed_importance = forest.feature_importances_
    result: list[tuple[str, float]] = []
    index = 0

    for feature in numeric_features:
        result.append((feature, float(transformed_importance[index])))
        index += 1

    encoder = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["onehot"]
        if categorical_features
        else None
    )
    if encoder is not None:
        for feature, categories in zip(categorical_features, encoder.categories_):
            width = len(categories)
            result.append(
                (feature, float(transformed_importance[index : index + width].sum()))
            )
            index += width

    return (
        pd.DataFrame(result, columns=["Feature", "Importance"])
        .sort_values("Importance", ascending=True)
        .reset_index(drop=True)
    )


def train_and_evaluate(
    frame: pd.DataFrame,
    features: Sequence[str],
    n_estimators: int = 350,
    max_depth: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate on a held-out split, then fit the persisted model on all rows."""
    selected = list(features)
    if not selected:
        raise ValueError("Select at least one feature before training.")
    if len(selected) != len(set(selected)):
        raise ValueError("Feature selections must not contain duplicates.")
    missing_features = [feature for feature in selected if feature not in frame.columns]
    if missing_features:
        raise ValueError(f"Selected features are missing from the data: {missing_features}.")
    if "Price" not in frame.columns:
        raise ValueError("The dataset must include the numeric Price target.")
    if n_estimators < 50 or n_estimators > 1_000:
        raise ValueError("The number of trees must be between 50 and 1,000.")
    if max_depth is not None and max_depth < 1:
        raise ValueError("Maximum tree depth must be at least 1.")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("The validation fraction must be between 0.10 and 0.40.")

    cleaned = frame.loc[:, [*selected, "Price"]].replace([np.inf, -np.inf], np.nan)
    if cleaned["Price"].isna().any():
        raise ValueError("Price contains missing or non-finite values.")

    numeric_features = [
        feature
        for feature in selected
        if pd.api.types.is_numeric_dtype(cleaned[feature].dtype)
        and not pd.api.types.is_bool_dtype(cleaned[feature].dtype)
    ]
    categorical_features = [
        feature for feature in selected if feature not in numeric_features
    ]
    empty_features = [feature for feature in selected if cleaned[feature].isna().all()]
    if empty_features:
        raise ValueError(f"Selected features contain no usable values: {empty_features}.")

    validation_rows = max(2, int(np.ceil(len(cleaned) * test_size)))
    if len(cleaned) - validation_rows < 2:
        raise ValueError("Not enough rows remain for both training and validation.")

    row_indices = np.arange(len(cleaned))
    train_indices, validation_indices = train_test_split(
        row_indices,
        test_size=validation_rows,
        random_state=random_state,
    )
    x = cleaned.loc[:, selected]
    y = cleaned["Price"]

    evaluation_pipeline = build_pipeline(
        numeric_features,
        categorical_features,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    evaluation_pipeline.fit(x.iloc[train_indices], y.iloc[train_indices])
    predictions = evaluation_pipeline.predict(x.iloc[validation_indices])
    actual = y.iloc[validation_indices]
    metrics = {
        "r2": float(r2_score(actual, predictions)),
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "test_fraction": float(len(validation_indices) / len(cleaned)),
    }

    final_pipeline = build_pipeline(
        numeric_features,
        categorical_features,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    final_pipeline.fit(x, y)
    importance = _feature_importance(
        final_pipeline, numeric_features, categorical_features
    )
    return {
        "pipeline": final_pipeline,
        "metrics": metrics,
        "feature_importance": importance,
        "features": selected,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }


def save_model(bundle: dict[str, Any], cache_key: str, path: Path = MODEL_PATH) -> None:
    """Atomically persist a fitted model bundle and its cache fingerprint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "cache_key": cache_key,
        "bundle": bundle,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".joblib.tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        joblib.dump(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_model(path: Path = MODEL_PATH) -> dict[str, Any] | None:
    """Load the persisted payload when available, otherwise return ``None``."""
    if not path.exists():
        return None
    payload = joblib.load(path)
    if payload.get("format_version") != MODEL_FORMAT_VERSION:
        return None
    return payload
