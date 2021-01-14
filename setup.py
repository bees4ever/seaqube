"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

# Following instructions from https://medium.com/better-programming/publishing-your-python-packages-to-pypi-e48c169f4f09

# Import our newly installed setuptools package.
import setuptools

# Opens our README.md and assigns it to long_description.
with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ['dill',
                'gensim',
                'googletrans==3.0.0',
                'nltk==3.5',
                'numpy',
                'pandas',
                'pathos==0.2.6',
                'progressbar2',
                'pyinflect==0.5.1',
                #'pytest==6.0.2',
                'schedule',
                'scikit-learn',
                'scipy',
                'sklearn',
                'spacy==2.3.2',
                'tqdm']

print("WELCOME TO SETUP SEAQUBE")
print(setuptools.find_packages())

# Function that takes several arguments. It assigns these values to our package.
setuptools.setup(
    # Distribution name the package. Name must be unique so adding your username at the end is common.
    name="seaqube",
    # Version number of your package. Semantic versioning is commonly used.
    version="0.1.1",
    # Author name.
    author="Benjamin Manns",
    # Author's email address.
    author_email="benjamin.manns@mni.thm.de",
    package_data={'seaqube.benchmark': ['benchmark_datasets/*.csv']},
    include_package_data=True,
    # Short description that will show on the PyPi page.
    description="Semantic Quality Benchmark for Word Embeddings, i.e. Natural Language Models in Python. The shortname is `SeaQuBe` or `seaqube`. Simple call it '| ˈsi: kjuːb |'",
    # Long description that will display on the PyPi page. Uses the repo's README.md to populate this.
    long_description=long_description,
    # Defines the content type that the long_description is using.
    long_description_content_type="text/markdown",
    # The URL that represents the homepage of the project. Most projects link to the repo.
    url="https://github.com/bees4ever/seaqube",
    # Finds all packages within in the project and combines them into the distribution together.
    packages=setuptools.find_packages(),
    # requirements or dependencies that will be installed alongside your package when the user installs it via pip.
    install_requires=requirements,
    # Gives pip some metadata about the package. Also displays on the PyPi page.
    classifiers=[
        "Programming Language :: Python :: 3.0",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # The version of Python that is required.
    python_requires='>=3',
)
