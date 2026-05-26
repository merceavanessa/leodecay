from setuptools import find_packages, setup

setup(
    name='leodecay',
    version='1.0',
    description='Quantifying and modeling the storm-induced orbital decay of low Earth orbit satellites .',
    author='Vanessa Mercea',
    license='MIT',
    packages=find_packages(exclude=["*.egg-info"])
)
