from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()

# Read the contents of requirements.txt file
with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="lobotomy",
    version="0.0.1",
    author="Aaron Klein",
    description="Two-stage weight sharing neural architecture search.",
    long_description=long_description,
    url="https://github.com/aaronkl/lobotomy",
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    keywords="machine learning" "weight sharing pruning LM NAS deep learning",
    packages=find_packages(
        include=[
            "lobotomy",
            "lobotomy.*",
        ]
    ),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.6",
    platforms=["Linux"],
    install_requires=required_packages,
    include_package_data=True,
    extras_require={
        "test": [
            "pytest"
        ],
        "all": [
            "pytest",
            "tqdm",
            "pygmo",
            "scikit-learn"
        ],
    },
)

