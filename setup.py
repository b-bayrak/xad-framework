from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="xad-framework",
    version="1.0.0",
    author="Betül Bayrak, Kerstin Bach",
    author_email="betul.bayrak@ntnu.no",
    description="Explainable anomaly detection: conformal confidence, "
                "uncertainty decomposition, and counterfactual explanations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/b-bayrak/xad-framework",
    license="CC-BY-NC-SA-4.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.7.0",
        "deap>=1.3.1",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
    },
)
