<p align="center">
    <img src="https://raw.githubusercontent.com/tatsushi-ikeda/libheom/master/etc/libheom_logo_simple.svg" alt="LibHEOM" height=96>
</p>

# LibHEOM: Library to Simulate Open Quantum Dynamics based on HEOM Theory

The current stable version is [v0.5](https://github.com/tatsushi-ikeda/libheom/tree/v0.5).

Version [1.0 (alpha)](https://github.com/tatsushi-ikeda/libheom/tree/develop) is under development.
The Master branch could be unstable.

## Introduction

`libheom` is a cross-platform, open-source library that supports open quantum dynamics simulations based on the hierarchical equations of motion (HEOM) theory.
This library provides low-level APIs to solve HEOM written C++17/CUDA.
Python 3 binding (`pylibheom`) and high-level APIs, including calculations of parameters of HEOM from specific spectral density models, are provided in [pyheom](https://github.com/tatsushi-ikeda/pyheom) package.

This library is still under development, and some optional functions are not implemented.
There are no guarantees about backward compatibility as of now (Version 1.0.0a1).

## TODO

-   Write API documentation
-   Rewrite codes for non-linear spectra calculations which are temporarily removed

## Required Packages

-   [CMake](https://cmake.org/):

    Cross-platform building system. Version 3.9 or later.

-   Python 3.6 or later and [Jinja2](https://github.com/pallets/jinja/):

    These are used to generate C++/CUDA codes from templates.

Additionally, at least one of the following is required as a part of linear algebra libraries.

-   Eigen3:
    [http://eigen.tuxfamily.org](http://eigen.tuxfamily.org/index.php?title=Main_Page)

-   Intel MKL:
    [https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)

-   CUDA &middot; cuBLAS &middot; cuSPARSE &middot; cuSOLVER:
    [https://developer.nvidia.com/cuda-zone](https://developer.nvidia.com/cuda-zone), [https://developer.nvidia.com/gpu-accelerated-libraries](https://developer.nvidia.com/gpu-accelerated-libraries)

## Installation

**CMake** is employed for cross-platform building. For details, see [INSTALL.md](INSTALL.md).

## Authors

-   **Tatsushi Ikeda** (ikeda.tatsushi.37u@kyoto-u.jp)

## Licence

[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](http://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_.28.22Revised_BSD_License.22.2C_.22New_BSD_License.22.2C_or_.22Modified_BSD_License.22.29)

`libheom` is distributed under the BSD 3-clause License. See the LICENSE.txt file for details.

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
   year = {2020},
   type = {Journal Article}
}
```

## Acknowledgments

<p align="center">
    <a href="https://www.jsps.go.jp/"><img src="https://www.jsps.go.jp/j-grantsinaid/06_jsps_info/g_120612/data/whiteKAKENHIlogoM_jp.jpg" alt="KAKENHI" height=48 hspace=8></a>
    <a href="https://www.moore.org/"><img src="https://www.moore.org/docs/default-source/Grantee-Resources/foundation-logos/moore-logo-color.jpg?sfvrsn=2" alt="MOORE" height=48 hspace=8></a>
</p>

-   A prototype of this library was developed for projects supported by [Japan Society for the Promotion of Science](https://www.jsps.go.jp/). 
    The current version is being developed for projects funded by JSPS again.
-   The version for the above research paper (v0.5) was developed in [the Scholes group](http://chemlabs.princeton.edu/scholes/) for projects supported by [the Gordon and Betty Moore Foundation](https://www.moore.org/).

