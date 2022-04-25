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
