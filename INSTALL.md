# Installation

## Build Binary

The automatic build system maintained by `libheom` is **CMake**.
The minimum version of CMake required is 3.0, but some external libraries may require a higher version of CMake.

First you need to install the dependent libraries, Eigen and pybind11.

* Eigen: You need to put the library into 3rdparty/ directory. Make sure that the location of the following file:
`3rdparty/Eigen/Eigen/Core`

* pybind11: You need to put the library into 3rdparty/ directory. Make sure that the location of the following file:
`3rdparty/pybind11/include/pybind11/pybind11.h`

These two libraries will be automatically placed by the following command:
```bash
git submodule update --init --recursive
```

### UNIX

Type the following commands from the source tree directory:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

Binaries and header files are located in the `build` and `build/include` directories, respectively.

### Windows (with Visual Studio VC++)

You can use **C++ CMake tools for Windows**, which is installed by default as a part of the Desktop development tool in Visual Studio.
You need to run the following commands from **Developer Command Prompt** of Visual Studio.

#### 1. Visual Studio IDE
Type the following commands from the source tree directory to generate Visual Studio solution (`.sln`) and projects (`.vcxproj`):

```bash
mkdir build
cd build
cmake ..
```

Then you can build the solution by using the Visual Studio IDE (Choose
`Release` build). Binaries and header files are located in the
`build\Release` and `build\include` directories, respectively.

#### 2. Command Line

If you want to use the cl compiler directly, type the following commands from the source tree directory:

```bash
mkdir build
cd build
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

Binaries and header files then are located in the `build` and `build\include` directories, respectively.

## Install Python Binding

After building the binaries, type the following commands from the binary directory (i.e., `build` or `build\Release`):

```bash
python3 setup.py install
```
