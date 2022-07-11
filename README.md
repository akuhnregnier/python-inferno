# python-inferno

Partly adapted from https://code.metoffice.gov.uk/svn/utils/inferno_py (joaoteixeira)

Use `git submodule update --init --recursive` to recursively update git submodules, e.g. the GPU code project which contains a submodule itself.

Install `py_gpu_inferno` using the following steps:
```sh
mkdir build
cd build
cmake ..
make py_gpu_inferno
make install
```

A separate miniconda installation was used for `rpy2` since it was found that installing `rpy2` broke `pyenv`.
This used the conda requirements file: `arm64_requirements_rpy2.yaml`.

See: https://mac.r-project.org/openmp/ regarding R installation on MAC M1.
Note also the following contents of `~/.R/Makevars`:
```
BLDSHARED=clang -bundle -undefined
CC=clang
CCBASE=clang
CLANG=clang
CLINKERBASE=clang
CPP=clang -E
CXX11=clang++
CXX14=clang++
CXX17=clang++
CXX20=clang++
CXX=clang++ -std=gnu++14
CXXBASE=clang++
CXXLINKERBASE=clang++
LDCXXSHARED=clang++ -bundle -undefined
LDSHARED=clang -bundle -undefined dynamic_lookup
LINKCC=clang
LTCC=clang
MAINCC=clang
OBJC=clang
OBJCXX=clang++
ccompiler=clang
cppcompiler=clang++
CPPFLAGS+=-Xclang -fopenmp
LDFLAGS+=-lomp
```
Note that `<miniconda_dir>/envs/python-inferno/lib/R/etc/Makeconf` and similar files were modified to use `clang`, `clang++` (instead of equivalent names prefixed like 'arm64-...' or similar), and `-fopenmp` -> `-Xclang -fopenmp`.

Run `install.packages(c("mgcv", "import", "purrr"), dependencies=TRUE)` to install R dependencies.

See also https://github.com/kharchenkolab/conos/wiki/Installing-Conos-for-Mac-OS.
