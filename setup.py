"""
Christopher Ondrusz
GitHub: acse_cro23
"""
from setuptools import setup

setup(
      name='FireDiff',
      version='1.0',
      description='Library for Wildfire Diffusion forecasting',
      author='Christopher Ondrusz',
      packages=['fireDiff'],
      package_dir={'fireDiff': 'fireDiff'},
      package_data={'fireDiff': ['*.csv', '*.txt']},
      include_package_data=True
      )
