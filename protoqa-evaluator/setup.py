"""
Evaluation framework for ProtoQA common sense QA dataset
"""
import fastentrypoints
from setuptools import find_packages, setup

setup(
    name="protoqa_evaluator",
    version="1.0",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={"": "src"},
    description="Evaluation scripts for ProtoQA common sense QA dataset.",
    install_requires=[
        "Click>=7.1.2",
        "scipy",
        "numpy",
        "nltk",
        "more-itertools",
        "xopen",
    ],
    extras_require={
        "test": ["pytest"],
        "crowdsource-conversion": ["pandas", "openpyxl"],
        "mlm-similarity": ["torch", "transformers", "scikit-learn"],
    },
    entry_points={
        "console_scripts": ["protoqa_evaluator = protoqa_evaluator.__main__:main"]
    },
)
