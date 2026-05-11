from setuptools import setup, find_packages

setup(
    name="votekv",
    version="0.1.0",
    description="VoteKV: GQA-aware SnapKV with Voting and Rescue Tokens",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "pyyaml>=6.0",
        "accelerate>=0.25.0",
        "datasets>=2.14.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
    ],
)
