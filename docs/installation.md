# Installation

## Requirements

- CMake 3.20+
- Python 3.9+ with Jinja2 (used to generate C++/CUDA code from templates)
- C++17 compiler (Intel icpx 2024+ or GCC 8+)

At least one linear algebra backend:

- **Eigen3** - included as a git submodule (`3rdparty/eigen`); default backend
- **Intel MKL** - install and activate before building
- **CUDA 11.7+** - requires nvcc and cuBLAS, cuSPARSE, cuSOLVER

## Getting the source

```bash
git clone https://github.com/tatsushi-ikeda/libheom
cd libheom
git submodule update --init --recursive
```

The submodule command fetches Eigen3 into `3rdparty/eigen/`.

## Building

### Eigen backend (default)

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

The static library `liblibheom.a` is generated in `build/src/`.

### MKL backend

```bash
mkdir build && cd build
cmake .. -DLIBHEOM_ENABLE_MKL=ON
cmake --build .
```

Ensure MKL is on `LD_LIBRARY_PATH` at runtime; otherwise import will fail with
`ImportError: libmkl_rt.so.1: cannot open shared object file`.

### CUDA backend

```bash
mkdir build && cd build
cmake .. -DLIBHEOM_ENABLE_CUDA=ON -DCUDA_ARCH_LIST=70
cmake --build .
```

Replace `70` with the compute capability of your GPU (e.g. `75` for T4, `80` for A100).

### As part of pyheom

libheom is normally used as a submodule of pyheom and built automatically by
`pip install -e .`. See the [pyheom installation guide](https://github.com/tatsushi-ikeda/pyheom/blob/master/docs/installation.md).

## CMake options

| Option | Description | Default |
|---|---|---|
| `CMAKE_CXX_COMPILER` | C++ compiler | system default |
| `CMAKE_BUILD_TYPE` | Build type | `Release` |
| `LIBHEOM_ENABLE_EIGEN` | Enable Eigen3 backend | `AUTO` |
| `LIBHEOM_ENABLE_MKL` | Enable Intel MKL backend | `AUTO` |
| `LIBHEOM_ENABLE_CUDA` | Enable CUDA backend | `AUTO` |
| `LIBHEOM_ENABLE_SINGLE` | Enable single-precision (complex64) | `OFF` |
| `LIBHEOM_ENABLE_DOUBLE` | Enable double-precision (complex128) | `ON` |
| `LIBHEOM_STACKTRACE` | Enable call-stack trace (slow, for debug) | `OFF` |
| `CUDA_ARCH_LIST` | CUDA compute capabilities, e.g. `70;80` | - |

`AUTO` detects the backend automatically; use `ON`/`OFF` to force enable or disable.
