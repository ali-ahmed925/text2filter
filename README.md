# text2filter

[![PyPI version](https://img.shields.io/pypi/v/text2filter)](https://pypi.org/project/text2filter/)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

`text2filter` is a Python package that allows you to apply image filters using **natural language phrases**. Instead of manually specifying filter parameters, you can simply describe the effect you want in words, and `text2filter` will automatically apply the corresponding image filter with appropriate intensity.

> **Note:** It is strongly recommended to use `text2filter` inside a **clean virtual environment** to avoid dependency conflicts, as it relies on specific versions of libraries like PyTorch, OpenCV, and transformers.

---

## Features

- Apply image filters using natural language phrases.
- Supports multiple filter types, including:
  - Gaussian filter
  - Median filter
  - Wavelet denoising
  - Bilateral filter
  - Boxblur filter
  - Cartoon filter
  - Gradient filter
  - HighBoost filter
  - Laplacian filter
  - NLM filter
  - Sobel filter
  - UnsharpMasking filter
  - (Extendable to more filters)
- Automatically selects **filter intensity** (low, medium, strong) based on your description.
- Easy-to-use API: single function call to apply filters.
- Pre-trained models included for language-to-filter mapping.

---

## Installation

You can install `text2filter` via PyPI (once published):

```bash
pip install text2filter
````

Or install the latest development version from your local repository:

```bash
git clone <https://github.com/ali-ahmed925/text2filter.git>
cd text2filter
python -m pip install -e .
```

> ⚠️ Recommended: Create a clean virtual environment before installing to avoid conflicts:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\Activate.ps1 # Windows PowerShell
```

---

## Dependencies

`text2filter` depends on the following packages (with recommended versions to avoid conflicts):

* `torch>=2.9.1,<3.0`
* `transformers>=4.57.1,<5`
* `sentence-transformers>=5.1.2,<6`
* `opencv-python>=4.12.0,<5`
* `numpy>=2.2.6,<3`
* `PyWavelets>=1.8.0,<2`
* `scikit-learn>=1.6.1,<2`

---

## Usage

### Import and Apply Filter

```python
from text2filter import apply_phrase_filter
import cv2

# Load image
img = cv2.imread("example.jpg")

# Apply filter based on natural language phrase
results = apply_phrase_filter(img, "reduce noise slightly")

# results is a list of (filtered_image, description)
for i, (img_out, desc) in enumerate(results):
    print(desc)
    cv2.imwrite(f"output_{i}.jpg", img_out)
```

### Output

For example, a phrase like `"reduce noise slightly"` might produce:

```
GaussianFilter | intensity=low | params={'ksize': [3, 3], 'sigmaX': 0.8}
GaussianFilter | intensity=medium | params={'ksize': [5, 5], 'sigmaX': 1.5}
GaussianFilter | intensity=strong | params={'ksize': [9, 9], 'sigmaX': 2.5}
```

The package automatically applies filters and returns processed images.

---

## Package Structure

```
text2filter/
│
├── text2filter/
│   ├── __init__.py
│   ├── apply_filter.py        # Core function to map phrases to filters
│   ├── filter_operations.py   # Actual implementations of image filters
│   ├── mappings.json          # Mapping of phrases to filters/intensities
│   └── model/                 # Pre-trained models for language understanding
│
├── small_run.py               # Example usage
├── setup.py
├── pyproject.toml
└── README.md
```

---

## Available Filters

* **Gaussian Filter** – smooths images using a Gaussian kernel.
* **Median Filter** – reduces salt-and-pepper noise.
* **Wavelet Denoise** – frequency-domain noise reduction using wavelets.
* ...

> Additional filters can be added by extending `filter_operations.py` and updating the `mappings.json`.

---

## Notes and Recommendations

* Always use a **clean virtual environment** to prevent version conflicts, especially with PyTorch, OpenCV, and transformers.
* Ensure images exist at the specified path; `cv2.imread` will fail if the file path is invalid.
* The package currently supports **Python 3.10+**.
* You can adjust filter intensity manually by modifying the `apply_phrase_filter` function, although the natural language interface handles this automatically.

---

## Contributing

Contributions are welcome! You can:

* Add new filters
* Improve phrase-to-filter mappings
* Optimize performance

Steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/my-feature`)
3. Make changes and commit (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a pull request

---

## License

MIT License © Ali Ahmed

---

## Acknowledgements

* OpenCV for image processing
* PyTorch and Transformers for language-to-filter mapping
* PyWavelets for wavelet denoising
* scikit-learn for encoding and preprocessing

---

## Contact

For questions or support, contact **Ali Ahmed** at `ali@example.com`.

```