from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.read().splitlines() #Split the Strings into List ['numpy','pandas']


setup(name = "mlops",
      version = "0.1",
      author = "jitesh",
      author_email = "jiteshvk13@gmail.com",
      description= "Airline Prediction",
      packages= find_packages(),
      install_requires = requirements)

