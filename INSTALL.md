# Installation

The automatic build system used in `libheom` is **CMake**.
The minimum version of CMake required is 3.9, but some external libraries may require a higher version of CMake.

## Preparation

First you need to install the dependent libraries, Eigen and pybind11.

* Eigen: You need to put the library into 3rdparty/ directory. Make sure that the location of the following file:
`3rdparty/Eigen/Eigen/Core`

* pybind11: You need to put the library into 3rdparty/ directory. Make sure that the location of the following file:
`3rdparty/pybind11/include/pybind11/pybind11.h`

These two libraries will be automatically placed by the following command:

```bash
git submodule update --init --recursive
```

## Install with python binding

Type the following command from the source tree directory:

```bash
pip install .
```

For developers, the following commands may be useful. 

```bash
cd /path/to/your/test/directory
python3 -m venv heom
source heom/bin/activate
pip install -e /path/to/libheom
pip install -e /path/to/pyheom
```

## Build only

Type the following commands from the source tree directory:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

Then binaries are generated in the `build/src` directory.
