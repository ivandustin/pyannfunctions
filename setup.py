from setuptools import setup, find_packages

setup(
    name='pyannfunctions',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'jaxlib',
        'jax'
    ]
)
