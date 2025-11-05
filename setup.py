import os
from setuptools import setup, find_packages

# Get the absolute path to the 'sam2' submodule directory
here = os.path.abspath(os.path.dirname(__file__))
sam2_path = os.path.join(here, "sam2")

# Check if the submodule path exists
if not os.path.exists(sam2_path) or not os.listdir(sam2_path):
    print("="*80)
    print("WARNING: The 'sam2' submodule is not initialized or is empty.")
    print("Please run: git submodule update --init --recursive")
    print("="*80)

setup(
    name="sam2animal",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # This tells pip to install the 'sam2' package
        # from the local 'sam2' directory.
        f"sam-2 @ file://{sam2_path}", 
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "matplotlib",
        "scipy",
        "transformers",
        "opencv-python",
        "pycocotools",
        "torchmetrics"
    ],
    description="Multi-animal tracking with SAM2.",
)