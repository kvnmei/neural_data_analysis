import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="neural-data-analysis",
    version="0.0.1",
    author="Kevin Mei",
    author_email="kmei@caltech.edu",
    description="Neural data analysis tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kvnmei/neural_data_analysis",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        # "PIL",
        "matplotlib",
        "seaborn",
    ],
    include_package_data=True,
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
