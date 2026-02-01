from setuptools import find_packages,setup


def get_requirements(filename):
    with open(filename) as f:
        packages_list = f.read().splitlines()
        
    return packages_list




setup(
    name= "First ML project",
    version="0.0.1",
    author="Prateek Patel",
    author_email="prateekpatel019@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements("requirement.txt")
)