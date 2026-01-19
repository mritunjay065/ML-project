from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> list[str]:
    """This function will return the list of requirements"""
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='MLProject',
    version='0.1.0',
    author='Mritunjay Singh',
    author_email='harshthaaur2005@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)