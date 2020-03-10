#!/usr/bin/env python

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='dsgd',
    version='0.2',
    author='Sergio Peñafiel',
    description='Tabular interpretable classifier based on Dempster-Shafer Theory and Gradient Descent',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Sergio-P/DSGD',
    packages=['dsgd'],
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'torch',
        'dill'
    ]
)
