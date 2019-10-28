import setuptools

d = {}
exec(open("neuronmi/version.py").read(), None, d)
version = d['version']
pkg_name = "neuronmi"
long_description = open("README.md").read()

setuptools.setup(
    name=pkg_name,
    version=version,
    author="Miroslav Kuchta, Alessio Paolo Buccino",
    author_email="alessiop.buccino@gmail.com",
    description="Python module for FEM simulation of neuronal activity",
    url="https://github.com/MiroK/nEuronMI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={},
    install_requires=[
        'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]
)
