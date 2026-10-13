"""Customer-facing house price estimator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from data_handler import DatasetValidationError, available_features, load_dataset
from model_engine import load_model, make_cache_key, save_model, train_and_evaluate


PUBLIC_DIR = Path(__file__).resolve().parent
CSS_PATH = PUBLIC_DIR / "custom_style.css"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_LABELS = {
    "area": "Property area",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "stories": "Stories",
    "mainroad": "Main-road access",
    "guestroom": "Guest room",
    "basement": "Basement",
    "hotwaterheating": "Hot-water heating",
    "airconditioning": "Air conditioning",
    "parking": "Parking spaces",
    "prefarea": "Preferred area",
    "furnishingstatus": "Furnishing status",
}

FEATURE_DESCRIPTIONS = {
    "area": "Choose the home's area from the range in the available listings.",
    "bedrooms": "Select the number of bedrooms.",
    "bathrooms": "Select the number of bathrooms.",
    "stories": "Select the number of floors in the home.",
    "mainroad": "Indicate whether the home has access to a main road.",
    "guestroom": "Indicate whether the home includes a guest room.",
    "basement": "Indicate whether the home has a basement.",
    "hotwaterheating": "Indicate whether hot-water heating is available.",
    "airconditioning": "Indicate whether air conditioning is available.",
    "parking": "Select the number of parking spaces.",
    "prefarea": "Indicate whether the home is in a preferred area.",
    "furnishingstatus": "Choose the home's furnishing level.",
}

SIZE_FEATURES = ("area", "bedrooms", "bathrooms", "stories", "parking")
HOME_FEATURES = (
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
)

st.set_page_config(
    page_title="Hearth | Home Price Estimate",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _apply_styles() -> None:
    """Load the local stylesheet for the customer-facing page."""
    try:
        stylesheet = CSS_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        st.error(f"The page could not load its display styles: {exc}")
        st.stop()
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)


def _feature_label(feature: str) -> str:
    """Return a friendly display name while retaining the workbook column key."""
    return FEATURE_LABELS.get(feature.casefold(), feature.replace("_", " ").title())


def _option_label(value: Any) -> str:
    """Display workbook category values with readable capitalization."""
    text = str(value)
    return text[:1].upper() + text[1:].replace("-", " ").replace("_", " ")


def _get_estimator(frame: pd.DataFrame) -> dict[str, Any]:
    """Load the matching saved estimator or refresh it without exposing ML controls."""
    payload = load_model()
    if payload is not None:
        bundle = payload["bundle"]
        features = bundle.get("features", [])
        n_estimators = bundle.get("n_estimators", 350)
        max_depth = bundle.get("max_depth")
        expected_key = make_cache_key(frame, features, n_estimators, max_depth)
        if (
            features
            and all(feature in frame.columns for feature in features)
            and payload.get("cache_key") == expected_key
        ):
            return bundle

    features = available_features(frame)
    bundle = train_and_evaluate(
        frame,
        features,
        n_estimators=350,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
    )
    cache_key = make_cache_key(
        frame,
        features,
        bundle["n_estimators"],
        bundle["max_depth"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    save_model(bundle, cache_key)
    return bundle


def _numeric_control(
    feature: str,
    series: pd.Series,
    column: Any,
) -> int | float:
    """Render small discrete values as labeled choices and ranges as sliders."""
    values = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    label = _feature_label(feature)
    key = f"house_{feature}"

    if values.empty:
        column.number_input(label, value=0.0, disabled=True, key=key)
        column.caption(FEATURE_DESCRIPTIONS.get(feature, ""))
        return 0.0

    minimum, maximum = float(values.min()), float(values.max())
    median = float(values.median())
    is_integer = pd.api.types.is_integer_dtype(series.dtype)
    if minimum == maximum:
        if is_integer:
            value: int | float = int(minimum)
            column.selectbox(label, [value], format_func=_option_label, key=key)
        else:
            value = minimum
            column.number_input(
                label, min_value=minimum, max_value=maximum, value=minimum, key=key
            )
        column.caption(FEATURE_DESCRIPTIONS.get(feature, ""))
        return value

    if is_integer:
        options = sorted({int(value) for value in values.unique()})
        if len(options) <= 12:
            default = min(options, key=lambda value: abs(value - median))
            value = column.selectbox(
                label,
                options,
                index=options.index(default),
                format_func=_option_label,
                key=key,
            )
            column.caption(FEATURE_DESCRIPTIONS.get(feature, ""))
            return int(value)

        low, high = int(np.floor(minimum)), int(np.ceil(maximum))
        default = int(np.clip(round(median), low, high))
        value = column.slider(
            label, min_value=low, max_value=high, value=default, key=key
        )
        column.caption(FEATURE_DESCRIPTIONS.get(feature, ""))
        return int(value)

    step = max((maximum - minimum) / 100, np.finfo(float).eps)
    aligned_median = minimum + round((median - minimum) / step) * step
    value = column.slider(
        label,
        min_value=minimum,
        max_value=maximum,
        value=float(np.clip(aligned_median, minimum, maximum)),
        step=step,
        key=key,
    )
    column.caption(FEATURE_DESCRIPTIONS.get(feature, ""))
    return float(value)


def _categorical_control(
    feature: str,
    series: pd.Series,
    column: Any,
) -> str:
    """Render observed workbook categories under a clear property-feature label."""
    observed = series.dropna().astype(str)
    options = sorted(observed.unique().tolist())
    label = _feature_label(feature)
    if not options:
        column.warning(f"No choices are available for {label.lower()}.")
        return ""

    modes = observed.mode()
    default = str(modes.iloc[0]) if not modes.empty else options[0]
    value = column.selectbox(
        label,
        options,
        index=options.index(default),
        format_func=_option_label,
        key=f"house_{feature}",
    )
    column.caption(FEATURE_DESCRIPTIONS.get(feature, ""))
    return str(value)


def _render_feature_group(
    frame: pd.DataFrame,
    features: list[str],
    numeric_features: list[str],
) -> dict[str, Any]:
    """Render each selected house feature with its user-facing label."""
    inputs: dict[str, Any] = {}
    columns = st.columns(3, gap="large")
    for index, feature in enumerate(features):
        column = columns[index % len(columns)]
        if feature in numeric_features:
            inputs[feature] = _numeric_control(feature, frame[feature], column)
        else:
            inputs[feature] = _categorical_control(feature, frame[feature], column)
    return inputs


def main() -> None:
    _apply_styles()
    st.markdown(
        '<div class="hero-kicker">A CLEARER WAY HOME</div>'
        "<h1>What could your home<br>be worth?</h1>"
        '<p class="hero-subtitle">Tell us a little about the property. We’ll give you '
        "a data-based estimate in moments.</p>",
        unsafe_allow_html=True,
    )

    try:
        frame, _ = load_dataset()
        bundle = _get_estimator(frame)
    except (DatasetValidationError, OSError, EOFError, ValueError, KeyError) as exc:
        st.error(f"The home estimator is temporarily unavailable: {exc}")
        st.stop()

    features = bundle["features"]
    numeric_features = bundle["numeric_features"]
    if not features:
        st.error("A home estimate is unavailable because property details are missing.")
        st.stop()

    with st.form("home_details"):
        with st.container(border=True):
            st.markdown("### The home")
            st.caption("Start with its size and layout.")
            size_features = [
                feature for feature in SIZE_FEATURES if feature in features
            ]
            remaining_features = [
                feature
                for feature in features
                if feature not in size_features and feature not in HOME_FEATURES
            ]
            property_values = _render_feature_group(
                frame, size_features + remaining_features, numeric_features
            )

        home_features = [
            feature for feature in HOME_FEATURES if feature in features
        ]
        if home_features:
            with st.container(border=True):
                st.markdown("### Home features")
                st.caption("Choose the features and amenities that match the property.")
                property_values.update(
                    _render_feature_group(frame, home_features, numeric_features)
                )

        submitted = st.form_submit_button(
            "Estimate my home’s value",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        prediction_frame = pd.DataFrame(
            [{feature: property_values[feature] for feature in features}],
            columns=features,
        )
        try:
            estimate = max(
                0.0, float(bundle["pipeline"].predict(prediction_frame)[0])
            )
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"We couldn’t estimate this property with those details: {exc}")
            st.stop()

        st.markdown(
            '<div class="prediction-card">'
            '<div class="prediction-label">Your estimated home value</div>'
            f'<div class="prediction-value">{estimate:,.0f}</div>'
            '<div class="prediction-note">Estimate shown in the same units as the '
            "provided property data.</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="estimate-hint">Your estimate will appear here after you '
            "enter the property details.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p class="disclaimer">This estimate is for guidance only and is not a '
        "formal appraisal or offer.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
#hide streamlit header footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
