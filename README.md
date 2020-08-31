# LibHEOM: Library to Simulate Open Quantum Dynamics based on HEOM Theory

## Introduction

`libheom` is a cross-platform, open-source library that supports open
quantum dynamics simulations based on the hierarchical equations of
motion (HEOM) theory.  This library provides low-level API to solve
HEOM written in C++11/CUDA and its Python 3 binding (`pylibheom`).
High-level API, including calculations of parameters of HEOM from
specific spectral density models, is provided in **pyheom** package.

This library is still under development, and some optional functions are not implemented.
The up-to-date code may be found on GitHub.
There are no guarantees about backwards compatibility as of now (Version 0.5).


## Required Packages
- Eigen: 
  [http://eigen.tuxfamily.org](http://eigen.tuxfamily.org/index.php?title=Main_Page)

- pybind11:
  [https://github.com/pybind/pybind11](https://github.com/pybind/pybind11)
- numpy:
  [https://numpy.org/](https://numpy.org/)

## Optional Dependencies
- Compute Unified Device Architecture (CUDA) : [https://developer.nvidia.com/cuda-zone](https://developer.nvidia.com/cuda-zone)
- cuBLAS &middot; cuSPARSE : [https://developer.nvidia.com/gpu-accelerated-libraries](https://developer.nvidia.com/gpu-accelerated-libraries)

## Installation

**CMake** is employed for cross-platform building. For details, see
[INSTALL.md](INSTALL.md).

## Authors
* **Tatsushi Ikeda** (ikeda.tatsushi.37u@kyoto-u.jp)

## Licence
[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](http://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_.28.22Revised_BSD_License.22.2C_.22New_BSD_License.22.2C_or_.22Modified_BSD_License.22.29)

`libheom` is distributed under the BSD 3-clause License. See the LICENSE.txt file for details.

## Acknowledgments
<p align="center">
<a href="https://www.moore.org/"><img src="https://numpy.org/_static/moore.svg" alt="MOORE" height=48 hspace=8></a>
<a href="https://www.jsps.go.jp/"><img src="https://www.jsps.go.jp/j-grantsinaid/06_jsps_info/g_120612/data/whiteKAKENHIlogoM_jp.jpg" alt="KAKENHI" height=48 hspace=8></a>
</p>


* This library has been developed and used in [the Scholes group](http://chemlabs.princeton.edu/scholes/) for projects supported by [the Gordon and Betty Moore Foundation](https://www.moore.org/).
* A prototype of this library was developed for projects supported by [Japan Society for the Promotion of Science](https://www.jsps.go.jp/).


****