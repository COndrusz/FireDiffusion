"""
Christopher Ondrusz
GitHub: acse_cro23
"""
from setuptools import setup

setup(
      name='FireDiffusion',
      version='1.0',
      description='Library for Wildfire Diffusion forecasting',
      author='Christopher Ondrusz',
      packages=['fireDiffusion'],
      package_dir={'fireDiffusion': 'fireDiffusion'},
      package_data={'fireDiffusion': ['*.csv', '*.txt']},
      include_package_data=True
      )
