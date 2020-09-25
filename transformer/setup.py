"""
This file contains the required information for setting up
the AI-platform environment. Any additional packages that are not preinstalled
on AI-platform should be specified in the REQUIRED_PACKAGES list.
"""

from setuptools import find_packages
from setuptools import setup

# Any special required packages
REQUIRED_PACKAGES = ['tensorflow==2.1.2']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='AI-platform boilerplate'
)
