from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.config import MODEL_PATH, SUPPORTED_IMAGE_TYPES
from src.inference import SceneClassifier


st.set_page_config(
    page_title="Indoor Scene Recognition",
    page_icon="🏢",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading Indoor-67 model...")
def load_classifier() -> SceneClassifier:
    """Load the model once and reuse it across Streamlit reruns."""
    return SceneClassifier(model_path=MODEL_PATH)


def format_label(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").title()


def main() -> None:
    st.title("Indoor Scene Recognition")

    st.write(
        "Upload an indoor image and the PyTorch model will classify "
        "it into one of the MIT Indoor-67 scene categories."
    )

    with st.sidebar:
        st.header("Model information")
        st.write("**Dataset:** MIT Indoor-67")
        st.write("**Framework:** PyTorch")
        st.write("**Architecture:** Efficient student CNN")
        top_k = st.slider(
            "Number of predictions",
            min_value=1,
            max_value=10,
            value=5,
        )

    try:
        classifier = load_classifier()
    except Exception as exc:
        st.error(f"Unable to load the model: {exc}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload an indoor scene",
        type=SUPPORTED_IMAGE_TYPES,
        help="Supported formats: JPG, JPEG, PNG and WebP.",
    )

    if uploaded_file is None:
        st.info("Upload an image to generate a prediction.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file could not be read as an image.")
        return

    image_column, result_column = st.columns([1, 1.4])

    with image_column:
        st.subheader("Uploaded Image")
        st.image(
            image,
            width=380,        # Change this value to 350, 400, 450 as you prefer
            caption="Uploaded Image",
        )
    with result_column:
        st.subheader("Prediction")

        with st.spinner("Classifying scene..."):
            try:
                predictions = classifier.predict(
                    image=image,
                    top_k=top_k,
                )
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                return

        best_prediction = predictions[0]

        st.metric(
            label="Predicted scene",
            value=format_label(best_prediction.class_name),
            delta=f"{best_prediction.confidence:.2%} confidence",
        )

        results = pd.DataFrame(
            {
                "Scene": [
                    format_label(prediction.class_name)
                    for prediction in predictions
                ],
                "Confidence": [
                    prediction.confidence
                    for prediction in predictions
                ],
            }
        )

        st.dataframe(
            results.style.format(
                {"Confidence": "{:.2%}"}
            ),
            hide_index=True,
            use_container_width=True,
        )

        chart_data = results.set_index("Scene")
        st.bar_chart(chart_data)

    with st.expander("Technical details"):
        st.write(f"Device: `{classifier.device}`")
        st.write(f"Input resolution: `{classifier.image_size} × {classifier.image_size}`")
        st.write(f"Number of classes: `{len(classifier.class_names)}`")


if __name__ == "__main__":
    main()