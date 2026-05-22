from setuptools import find_packages, setup

setup(
    name='leo_orbital_decay_rate_impacts',
    version='1.0',
    description='Quantifying and modeling the storm-induced orbital decay of low Earth orbit satellites .',
    author='Vanessa Mercea',
    license='MIT',
    packages=find_packages(include=["leo_orbital_decay_rate_impacts*"])
)
