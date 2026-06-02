<p align="center">
    <img src="https://raw.githubusercontent.com/tatsushi-ikeda/libheom/master/etc/libheom_logo_simple.svg" alt="LibHEOM" height=96>
</p>

# LibHEOM: C++ Library for Open Quantum Dynamics based on HEOM Theory

`libheom` is a cross-platform, open-source C++17/CUDA library for open quantum
dynamics simulations based on the hierarchical equations of motion (HEOM) theory.
It provides CPU (Eigen/MKL) and GPU (CUDA) backends.

Python 3 bindings and high-level APIs (spectral density models, noise decomposition,
automatic parameter tuning) are provided in [pyheom](https://github.com/tatsushi-ikeda/pyheom).

The current release is v1.0.0b1.

## Documentation

Full documentation is available in [`docs/`](docs/index.md).

## Requirements

- CMake 3.20+
- Python 3.9+ with Jinja2 (for C++ code generation from templates)
- C++17 compiler (Intel icpx 2024+ or GCC 8+)

At least one linear algebra backend:

- Eigen3 (included as a submodule; default)
- Intel MKL
- CUDA 11.7+ with cuBLAS, cuSPARSE, cuSOLVER

## Installation

Initialize submodules and build:

```bash
git submodule update --init --recursive
mkdir build && cd build
cmake ..
cmake --build .
```

For MKL, CUDA, and all CMake options see [`docs/installation.md`](docs/installation.md).

## Authors

- **Tatsushi Ikeda** (ikeda.tatsushi.37u@kyoto-u.jp)

## License

[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](http://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_.28.22Revised_BSD_License.22.2C_.22New_BSD_License.22.2C_or_.22Modified_BSD_License.22.29)

`libheom` is distributed under the BSD 3-clause License. See the `LICENSE.txt` file for details.

## Citation Information

```Plain Text
@article{ikeda2020jcp,
   author = {Ikeda, Tatsushi and Scholes, Gregory D.},
   title = {Generalization of the hierarchical equations of motion theory for efficient calculations with arbitrary correlation functions},
   journal = {The Journal of Chemical Physics},
   volume = {152},
   number = {20},
   pages = {204101},
   ISSN = {0021-9606},
   DOI = {10.1063/5.0007327},
   url = {https://doi.org/10.1063/5.0007327},
   eprint = {2003.06134},
   archivePrefix = {arXiv},
   year = {2020},
   type = {Journal Article}
}
```

## Acknowledgments

<p align="center">
    <a href="https://www.jsps.go.jp/"><img src="https://www.jsps.go.jp/j-grantsinaid/img/logo/KAKENHIlogo_M.jpg" alt="KAKENHI" height=48 hspace=8></a>
    <a href="https://www.moore.org/"><img src="https://www.moore.org/docs/default-source/Grantee-Resources/foundation-logos/moore-logo-color.jpg?sfvrsn=2" alt="MOORE" height=48 hspace=8></a>
</p>

- A prototype of this library was developed for projects supported by [Japan Society for the Promotion of Science](https://www.jsps.go.jp/).
  The current version is being developed for projects funded by JSPS again.
- The version for the above research paper (v0.5) was developed in [the Scholes group](http://chemlabs.princeton.edu/scholes/) for projects supported by [the Gordon and Betty Moore Foundation](https://www.moore.org/).
