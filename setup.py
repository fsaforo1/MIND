import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mimind",
    version="1.0.0",
  	zip_safe=False,
  	license="GPLv3",
  	author="Dr. Yves-Laurent Kom Samo",
  	author_email="github@kxy.ai",
  	description = "Inductive Mutual Information Estimation, A Convex Maximum-Entropy Copula Approach",
  	long_description=long_description,
  	long_description_content_type='text/markdown',  # This is important
  	keywords = ["Mutual Information", "Data Valuation"],
    url='https://github.com/fsaforo1/MIND',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.13.1", "scipy>=1.4.1", "pandas>=0.23.0", "pandarallel", "ipywidgets", "scikit-learn"],
) 
