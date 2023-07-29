#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Ajala Marvellous",
    author_email='ajalaoluwamayowa00@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="This is a simple deep learning framework for many deep learning algorithms built from scratch with just numpy, it is meant for educational purposes only",
    entry_points={
        'console_scripts': [
            'dl_framework=dl_framework.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='dl_framework',
    name='dl_framework',
    packages=find_packages(include=['dl_framework', 'dl_framework.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ajalamarvellous/dl_framework',
    version='0.1.0',
    zip_safe=False,
)
