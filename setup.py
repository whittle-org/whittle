from setuptools import setup, find_packages

setup(
    name="lobotomy",
    version="0.0.1",
    description="two-stage weight sharing neural architecture search",
    author="Aaron Klein",
    packages=find_packages(
        include=[
            "lobotomy",
            "lobotomy.*",
        ]
    ),
)
