#!/usr/bin/env python

"""The setup script."""
from os import path
from setuptools import setup, find_packages
import sys
import versioneer

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]

with open(path.join(here, 'requirements-dev.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    dev = [line for line in requirements_file.read().splitlines()
           if not line.startswith('#')]

with open(path.join(here, 'requirements-docs.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    docs = [line for line in requirements_file.read().splitlines()
            if not line.startswith('#')]

setup(
    author="Zineb Sordo, Daniela Ushizima, Peter Andeer, James Sethian and Trent Northen.",
    author_email='zsordo@lbl.gov',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    description="Segmentation pipeline for EcoFAB images",
    entry_points={
        'console_scripts': [
            'train_rhizonet=rhizonet.train:main',
            'predict_rhizonet=rhizonet.predict:main',
            'postprocess_rhizonet=rhizonet.postprocessing:main',
            'patchify_rhizonet=rhizonet.prepare_patches:main',
            'evalmetrics_rhizonet=rhizonet.metrics:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='rhizonet',
    name='rhizonet',
    packages=find_packages(include=['rhizonet', 'rhizonet.*']),
    test_suite='tests',
    url='https://github.com/lbl-camera/rhizonet',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
    extras_require={
        'tests': ['pytest', 'codecov', 'pytest-cov'],
        'docs': ['sphinx', 'sphinx-rtd-theme', 'myst-parser', 'myst-nb', 'sphinx-panels', 'autodocs']
    },
    setup_requires=["wheel"]
)
