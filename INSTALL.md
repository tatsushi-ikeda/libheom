# Installation

## Preparation

First, you need to install the compulsory and optional dependent libraries.

### Compulsory package

```bash
pip3 install jinja2
```

### Optional package

At least one of the following is required as a part of linear algebra libraries.

- **Eigen3**

Install Eigen3 or put the library into `3rdparty/` directory.
In the case of the latter, make sure that the location of the following file: `3rdparty/eigen/Eigen/Core`
This will be automatically achieved by the following command:

```bash
git submodule update --init --recursive
```

- **Intel MKL**

Install Intel MKL and make sure that C++ compiler can detect `mkl.h`.

- **CUDA**

Install CUDA and make sure that `nvcc` is executable.

## Installation

### As a part of `pyheom`

While it is possible to use `libheom` alone (some examples are coming soon), we recommend to use `libheom` as a part of `pyheom`. 
For installation with `pyheom`, see `INSTALL.md` in `pyheom`.

### Build C++ part alone

Type the following commands from the source tree directory:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

Then binary `liblibheom.a` is generated in the `build/src` directory.

## CMake Options

| name                    | meaning                                       | values                                             |
|-------------------------|-----------------------------------------------|----------------------------------------------------|
| `CMAKE_CXX_COMPILER`    | C++ compiler                                  |                                                    |
| `CMAKE_BUILD_TYPE`      | Build type                                    | `Release`* or `Debug`                              |
| `LIBHEOM_ENABLE_EIGEN`  | Enable Eigen 3 module                         | `AUTO`*, `ON`, or `OFF`                            |
| `LIBHEOM_ENABLE_MKL`    | Enable Intel MKL module                       | `AUTO`*, `ON`, or `OFF`                            |
| `LIBHEOM_ENABLE_CUDA`   | Enable CUDA module                            | `AUTO`*, `ON`, or `OFF`                            |
| `LIBHEOM_ENABLE_SINGLE` | Enable single-precision floating-point format | `ON` or `OFF`*                                     |
| `LIBHEOM_ENABLE_DOUBLE` | Enable double-precision floating-point format | `ON`* or `OFF`                                     |
| `LIBHEOM_STACKTRACE`    | Enable call stack trace for debug (slow)      | `ON` or `OFF`*                                     |
| `CUDA_ARCH_LIST`        | CUDA architectures for built binary           | semicolon-separated list of numbers, e.g., `60;70` |


