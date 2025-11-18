# apply_filter.py
import os
import json
import logging
from copy import deepcopy
import pickle
from .filter_operations import *
from .embeddings_generator import Predictor

# ----------------------
# Setup logger
# ----------------------
logging.basicConfig(level=logging.INFO, format='[text2filter] %(message)s')
logger = logging.getLogger(__name__)

# ----------------------
# Load filter mappings JSON
# ----------------------
with open(os.path.join(os.path.dirname(__file__), "mappings.json"), "r") as f:
    FILTER_JSON = json.load(f)

# ----------------------
# Global predictor cache
# ----------------------
_GLOBAL_PREDICTOR = None


def _get_global_predictor():
    """Lazy-load the predictor for the first time."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        logger.info("Loading model and encoders for first use...")

        MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
        # Load encoders
        with open(os.path.join(MODEL_DIR, "filter_encoder.pkl"), "rb") as f:
            filter_encoder = pickle.load(f)

        with open(os.path.join(MODEL_DIR, "intensity_encoder.pkl"), "rb") as f:
            intensity_encoder = pickle.load(f)

        model_path = os.path.join(MODEL_DIR, "twohead_model.pt")
        transformer_name = "sentence-transformers/all-MiniLM-L6-v2"

        _GLOBAL_PREDICTOR = Predictor(
            model_path=model_path,
            filter_encoder=filter_encoder,
            intensity_encoder=intensity_encoder,
            transformer_name=transformer_name
        )
        logger.info("Model loaded successfully.")

    return _GLOBAL_PREDICTOR


# ----------------------
# Main function
# ----------------------
def apply_phrase_filter(image, phrase, predictor=None, return_all_intensities=True):
    """
    Apply a predicted filter to an image based on a textual phrase.

    Returns 3 tuples (predicted intensity first, then other two):
    (image, description)

    Parameters:
    - image: np.array, input image
    - phrase: str, textual description of filter
    - predictor: optional, a Predictor object (will be lazy-loaded if None)
    - return_all_intensities: bool, if True returns all intensity variants
    """
    if predictor is None:
        predictor = _get_global_predictor()

    logger.info("Running prediction for phrase: '%s'", phrase)
    pred_filter, pred_intensity = predictor.predict(phrase)
    logger.info("Predicted filter: %s | intensity: %s", pred_filter, pred_intensity)

    # Get intensity mapping for this filter
    filter_data = FILTER_JSON[pred_filter]["intensity_mapping"]
    intensities = ["low", "medium", "strong"]
    ordered_intensities = [pred_intensity] + [i for i in intensities if i != pred_intensity]

    result = []

    for intensity in ordered_intensities:
        params = filter_data[intensity]
        library_info = FILTER_JSON[pred_filter].get("library", "custom")
        img_copy = deepcopy(image)

        logger.info("Applying %s | intensity=%s", pred_filter, intensity)

        # ----------------------
        # Dispatch filter functions
        # ----------------------
        if pred_filter == "MedianFilter":
            img_filtered = apply_median(img_copy, params["ksize"])
        elif pred_filter == "GaussianFilter":
            img_filtered = apply_gaussian(img_copy, params["ksize"], params["sigmaX"])
        elif pred_filter == "BoxBlurFilter":
            img_filtered = apply_boxblur(img_copy, params["ksize"])
        elif pred_filter == "BilateralFilter":
            img_filtered = apply_bilateral(img_copy, params["d"], params["sigmaColor"], params["sigmaSpace"])
        elif pred_filter == "SobelFilter":
            img_filtered = apply_sobel(img_copy, params["ksize"])
        elif pred_filter == "LaplacianSharpenFilter":
            img_filtered = apply_laplacian_sharpen(img_copy, params["ksize"])
        elif pred_filter == "HighBoostFilter":
            img_filtered = apply_highboost(img_copy, params["alpha"])
        elif pred_filter == "UnsharpMaskingFilter":
            img_filtered = apply_unsharp(img_copy, params["amount"], params["radius"])
        elif pred_filter == "CartoonEffectFilter":
            img_filtered = apply_cartoon(img_copy, params["bilateral_d"], params["sigmaColor"], params["sigmaSpace"],
                                         params["edges_ksize"])
        elif pred_filter == "NonLocalMeansFilter":
            img_filtered = apply_nlm(img_copy, params["h"], params["templateWindowSize"], params["searchWindowSize"])
        elif pred_filter == "WaveletDenoiseFilter":
            img_filtered = apply_wavelet_denoise(img_copy, params["wavelet"], params["level"],
                                                 params["threshold_scale"])
        elif pred_filter == "GradientFilter":
            img_filtered = apply_gradient(img_copy, params["ksize"])
        else:
            img_filtered = img_copy  # fallback

        desc = f"{pred_filter} | intensity={intensity} | params={params} | library={library_info}"
        result.append((img_filtered, desc))

        if not return_all_intensities:
            break

        logger.info("Filter applied successfully!")

    logger.info("All filters applied, returning results.")
    return tuple(result)
