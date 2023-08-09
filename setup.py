from setuptools import setup, find_packages

VERSION = '0.0.5'
DESCRIPTION = 'Super Learner Equation Modeling Package'
LONG_DESCRIPTION = 'Super Learner Equation Modeling for Causal Inference on DAGs with Machine Learning.'

# Setting up
setup(
    name="slem-learn",
    version=VERSION,
    author="Matthew J. Vowels",
    author_email="<etmat1000@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['sklearn', 'scipy', 'networkx', 'pandas', 'tqdm'],
    keywords=['python', 'statistics', 'causality', 'causal inference', 'machine learning', 'super learner', 'graphs', 'DAGs'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)