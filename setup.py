"""
To convert project into package.

"""

from setuptools import setup , find_packages

setup(
    name = "document_loader",
    author = "Bhavya Mandaliya",
    version = "0.1",
    description= "A Document management portal built with FastAPI." ,
    packages= find_packages()

)