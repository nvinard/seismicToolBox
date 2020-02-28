from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

#REQUIRED_PACKAGES = ['tqdm']

setup(
    name="seismicToolBox",
    version="0.0.11",
    packages=find_packages(),
    include_package_data=True,
    author="Nicolas Vinard, Musab al Hasani",
    author_email="n.a.vinard@tudelft.nl, m.m.k.alhasani@tudelft.nl",
    description="A python toolbox with basic stuff for seismics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nvinard/seismicToolBox",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = ['numpy', 'tqdm'],
)

