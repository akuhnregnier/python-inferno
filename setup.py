# -*- coding: utf-8 -*-
import os

import setuptools
from Cython.Build import cythonize

with open("README.md", "r") as f:
    readme = f.read()

os.environ["CFLAGS"] = "-O3 -ffast-math -fopenmp"
# os.environ['CFLAGS'] = '-O3 -ffast-math'

setuptools.setup(
    name="python_inferno",
    author="Alexander Kuhn-Regnier",
    author_email="ahf.kuhnregnier@gmail.com",
    description="Python version of INFERNO from JULES",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/akuhnregnier/python-inferno",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    setup_requires=["setuptools-scm", "cython"],
    use_scm_version=dict(write_to="src/python_inferno/_version.py"),
    include_package_data=True,
    ext_modules=cythonize(
        "src/python_inferno/cython/*.pyx",
    ),
)

# ext_modules = [
#     setuptools.Extension(
#         "cython_inferno",
#         ["src/python_inferno/cython/cython_inferno.pyx"],
#         extra_compile_args=["-fopenmp"],
#         extra_link_args=["-fopenmp"],
#     )
# ]
#
# setuptools.setup(
#     name="python_inferno",
#     author="Alexander Kuhn-Regnier",
#     author_email="ahf.kuhnregnier@gmail.com",
#     description="Python version of INFERNO from JULES",
#     long_description=readme,
#     long_description_content_type="text/markdown",
#     url="https://github.com/akuhnregnier/python-inferno",
#     packages=setuptools.find_packages("src"),
#     package_dir={"": "src"},
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "Operating System :: OS Independent",
#     ],
#     python_requires=">=3.7",
#     setup_requires=["setuptools-scm", "cython"],
#     use_scm_version=dict(write_to="src/python_inferno/_version.py"),
#     include_package_data=True,
#     ext_modules=cythonize(ext_modules),
# )
