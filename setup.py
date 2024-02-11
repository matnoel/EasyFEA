from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="EasyFEA",
    version="0.0.1",
    description="User-friendly Python library that simplifies finite element simulations",
    packages=find_packages(exclude=["tests*","results*","codes*"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matnoel/EasyFEA",
    author="matnoel",
    author_email="matthieu.noel7@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "scipy", "matplotlib", "pyvista", "gmsh >= 4.12.0", "numba", "pandas"],
    extras_require={
        "solvers": ["pypardiso", "petsc", "petsc4py"],
        "dev": ["pytest", "twine"],
    },
    python_requires="<3.11",
)