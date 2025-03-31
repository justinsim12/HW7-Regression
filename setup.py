import setuptools

setuptools.setup(
    name="hw7-Regression",
    version="0.1.0",
    author="Justin Sim",
    author_email="justin.sim@ucsf.edu",
    description="Logistic regression using gradient descent",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/justinsim12/HW7-Regression",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scikit-learn",
        "matplotlib",
        "pytest",
        "pandas"
    ],
)