"""Excel data loading, validation, and deterministic sample-data generation."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException


PUBLIC_DIR = Path(__file__).resolve().parent
DATA_DIR = PUBLIC_DIR / "data"
DEFAULT_DATA_PATH = DATA_DIR / "house_prices.xlsx"
PROVIDED_DATA_PATH = DATA_DIR / "Housing.xlsx"
TARGET_COLUMN = "Price"


class DatasetValidationError(ValueError):
    """Raised when an Excel dataset cannot be used for price prediction."""


def generate_sample_dataset(
    path: Path = DEFAULT_DATA_PATH,
    rows: int = 1_000,
    random_state: int = 42,
) -> Path:
    """Create a reproducible, realistic sample workbook at ``path``."""
    if rows < 20:
        raise ValueError("Sample data must contain at least 20 rows.")

    rng = np.random.default_rng(random_state)
    neighborhoods = np.array(
        ["Maple Grove", "Riverside", "Oak Hill", "Downtown", "Cedar Park"]
    )
    neighborhood_premium = {
        "Maple Grove": 42_000,
        "Riverside": 24_000,
        "Oak Hill": 58_000,
        "Downtown": 12_000,
        "Cedar Park": 34_000,
    }

    square_feet = np.clip(rng.normal(1_950, 620, rows), 650, 5_000).round().astype(int)
    bedrooms = np.clip(rng.poisson(3.2, rows) + 1, 1, 7)
    bathrooms = np.clip(rng.normal(2.2, 0.8, rows), 1, 5).round(1)
    neighborhood = rng.choice(
        neighborhoods, size=rows, p=[0.24, 0.18, 0.16, 0.20, 0.22]
    )
    year_built = rng.integers(1950, 2025, rows)
    garage_cars = np.clip(rng.poisson(1.5, rows), 0, 4)

    age = 2025 - year_built
    price = (
        48_000
        + square_feet * 142
        + bedrooms * 8_500
        + bathrooms * 17_000
        + np.array([neighborhood_premium[name] for name in neighborhood])
        + garage_cars * 13_000
        - np.maximum(age - 20, 0) * 420
        + rng.normal(0, 32_000, rows)
    )
    price = np.maximum(price, 55_000).round(-2).astype(int)

    frame = pd.DataFrame(
        {
            "Square_Feet": square_feet,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Neighborhood": neighborhood,
            "Year_Built": year_built,
            "Garage_Cars": garage_cars,
            TARGET_COLUMN: price,
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(path, index=False, engine="openpyxl")
    return path


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers and validate the required target and usable rows."""
    if frame.empty:
        raise DatasetValidationError("The Excel sheet contains no data rows.")

    cleaned = frame.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    if any(not column for column in cleaned.columns):
        raise DatasetValidationError("Column names must not be empty.")
    if cleaned.columns.duplicated().any():
        duplicates = cleaned.columns[cleaned.columns.duplicated()].tolist()
        raise DatasetValidationError(
            f"Column names must be unique; duplicates found: {duplicates}."
        )
    target_matches = [
        column for column in cleaned.columns if column.casefold() == TARGET_COLUMN.casefold()
    ]
    if len(target_matches) > 1:
        raise DatasetValidationError(
            f"Found multiple columns matching the '{TARGET_COLUMN}' target."
        )
    if not target_matches:
        raise DatasetValidationError(
            f"The workbook must include a numeric '{TARGET_COLUMN}' target column "
            "(column names are matched without regard to case)."
        )

    cleaned = cleaned.rename(columns={target_matches[0]: TARGET_COLUMN})
    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    if len(cleaned) < 10:
        raise DatasetValidationError(
            "At least 10 rows with a valid numeric Price are required for training."
        )
    if len([column for column in cleaned.columns if column != TARGET_COLUMN]) == 0:
        raise DatasetValidationError(
            "The workbook must include at least one feature column besides Price."
        )
    return cleaned


def load_dataset(uploaded_file: BinaryIO | None = None) -> tuple[pd.DataFrame, str]:
    """Load an uploaded Excel workbook or create/load the default sample workbook."""
    if uploaded_file is not None:
        try:
            frame = pd.read_excel(uploaded_file, engine="openpyxl")
        except (BadZipFile, InvalidFileException, OSError, ValueError, ImportError) as exc:
            raise DatasetValidationError(
                f"Could not read the uploaded workbook: {exc}"
            ) from exc
        source = getattr(uploaded_file, "name", "Uploaded workbook")
    else:
        data_path = (
            PROVIDED_DATA_PATH
            if PROVIDED_DATA_PATH.exists()
            else DEFAULT_DATA_PATH
        )
        if not data_path.exists():
            generate_sample_dataset()
        try:
            frame = pd.read_excel(data_path, engine="openpyxl")
        except (BadZipFile, InvalidFileException, OSError, ValueError, ImportError) as exc:
            raise DatasetValidationError(
                f"Could not read {data_path.name}: {exc}"
            ) from exc
        source = data_path.name

    return validate_dataset(frame), source


def available_features(frame: pd.DataFrame) -> list[str]:
    """Return non-target columns with at least one usable observation."""
    return [
        column
        for column in frame.columns
        if column != TARGET_COLUMN and frame[column].notna().any()
    ]
