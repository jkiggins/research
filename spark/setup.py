import setuptools
from setuptools import setup

from os import path

pwd = path.abspath(path.dirname(__file__))

setup(
    name="spark",
    version="0.0.1",
    description="A library for deep learning with spiking neural networks",
    url="",
    author="Jacob Kiggins",
    author_email="jkiggins@protonmail.com",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine learning spiking neural networks",
)
