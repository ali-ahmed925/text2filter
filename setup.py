from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    description = f.read()

setup(
    name="text2filter",
    version="0.2.0",
    description="Apply image filters using natural language phrases",
    author="Ali Ahmed",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.9.1,<3.0",
        "transformers>=4.57.1,<5",
        "sentence-transformers>=5.1.2,<6",
        "opencv-python>=4.12.0,<5",
        "numpy>=2.2.6,<3",
        "PyWavelets>=1.8.0,<2",
        "scikit-learn>=1.6.1,<2"
    ],
    long_description=description,
    long_description_content_type="text/markdown",
    package_data={
        "text2filter": ["mappings.json", "model/*"]
    },
    python_requires=">=3.10",
)
