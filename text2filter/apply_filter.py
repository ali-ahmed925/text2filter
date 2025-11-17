import json
from copy import deepcopy
from filter_operations import *

with open("filter_mappings.json","r") as f:
    FILTER_JSON = json.load(f)

def apply_phrase_filter(image, phrase, predictor, return_all_intensities=True):

    pred_filter, pred_intensity = predictor.predict(phrase)

    filter_data = FILTER_JSON[pred_filter]["intensity_mapping"]
    intensities = ["low","medium","strong"]
    ordered_intensities = [pred_intensity] + [i for i in intensities if i != pred_intensity]

    result = []

    for intensity in ordered_intensities:
        params = filter_data[intensity]
        library_info = FILTER_JSON[pred_filter].get("library", "custom")
        img_copy = deepcopy(image)


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
            img_filtered = apply_cartoon(img_copy, params["bilateral_d"], params["sigmaColor"], params["sigmaSpace"], params["edges_ksize"])
        elif pred_filter == "NonLocalMeansFilter":
            img_filtered = apply_nlm(img_copy, params["h"], params["templateWindowSize"], params["searchWindowSize"])
        elif pred_filter == "WaveletDenoiseFilter":
            img_filtered = apply_wavelet_denoise(img_copy, params["wavelet"], params["level"], params["threshold_scale"])
        elif pred_filter == "GradientFilter":
            img_filtered = apply_gradient(img_copy, params["ksize"])
        else:
            img_filtered = img_copy


        desc = f"{pred_filter} | intensity={intensity} | params={params} | library={library_info}"
        result.append((img_filtered, desc))

        if not return_all_intensities:
            break

    return tuple(result)
