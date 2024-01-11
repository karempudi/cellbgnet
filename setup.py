import os
import setuptools




from setuptools import setup

if os.environ["CONDA_DEFAULT_ENV"] in ["bgnet"]:
    # conda requirements are set using the .yml files in the 
    # conda_envs directory.
    requirements = []
else:
    # TODO:add pip requirements here, and make sure you can install 
    # using only pip later.
    requirements = []

setup(
    name='cellbgnet',
    version = '0.0.1',
    packages = setuptools.find_packages(),
    install_requires = requirements,
    entry_points= {
        'console_scripts' : [
            'cellbgnet.train = cellbgnet.train:main',
        ]
    },
    zip_safe=False,
    url='https://github.com/karempudi/cellbgnet',
    license='MIT',
    author='Praneeth Karempudi',
    author_email='praneeth.karempudi@gmail.com',
    description='CELLBGNET'
)
