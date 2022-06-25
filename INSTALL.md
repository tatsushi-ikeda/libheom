# Installation

The build system used in `libheom` is **CMake**.
The minimum version of CMake required is 3.9, but some external libraries may require a higher version of CMake.

## Preparation

First you need to install the optional dependent libraries.

For Eigen, you can put the library into 3rdparty/ directory
Make sure that the location of the following file: `3rdparty/eigen/Eigen/Core`

Eigen will be automatically placed by the following command:

```bash
git submodule update --init --recursive
```

## Build only

Type the following commands from the source tree directory:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

Then binary `liblibheom.a` is generated in the `build/src` directory.

## CMake Options

| name                   | meaning                                 | values                                             |
|------------------------|-----------------------------------------|----------------------------------------------------|
| `CMAKE_CXX_COMPILER`   | C++ compiler                            |                                                    |
| `CMAKE_BUILD_TYPE`     | Build type                              | `Release`* or `Debug`                              |
| `LIBHEOM_ENABLE_EIGEN` | Activate Eigen                          | `AUTO`*, `ON`, or `OFF`                            |
| `LIBHEOM_ENABLE_MKL`   | Activate Intel MKL                      | `AUTO`*, `ON`, or `OFF`                            |
| `LIBHEOM_ENABLE_CUDA`  | Activate CUDA                           | `AUTO`*, `ON`, or `OFF`                            |
| `CUDA_ARCH_LIST`       | Cuda architecture list for built binary | semicolon-separated list of numbers, e.g., `60;70` |


