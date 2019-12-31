# YATDFT: Yet Another Tiny DFT

A tiny library for constructing matrices used in Hartree-Fock (HF) and Kohn-Sham density functional theory (KS-DFT) using Gaussian basis sets. 

YATDFT requires:

* [Simint](https://github.com/simint-chem/simint-generator) for electron repulsion integrals (ERI)
* [Libxc](https://gitlab.com/libxc/libxc) for some XC GGA functionals (YATDFT has some built-in implementations)
* OpenMP and C99 supported C compiler
* Intel MKL (could be replace by other BLAS+LAPACK implementations, have not tested yet)

YATDFT can construct:

* Core Hamiltonian matrix
* Overlap matrix
* Basis transformation matrix
* Coulomb matrix
* HF exchange matrix
* CDIIS Pulay mixing for Fock matrix
* Density matrix (from Fock matrix or SAD initial guess)
* DFT exchange-correlation matrix
  * Built-in LDA XC functionals: Slater exchange, Slater Xalpha correlation (alpha = 0.7 - 2/3), PZ81 correlation , PW92 correlation  
  * Built-in GGA XC functionals: PBE exchange & correlation, B88 exchange, LYP correlation
  * GGA XC functionals from Libxc: PW91 exchange & correlation, G96 exchange, PW86 exchange, P86 correlation

## Compiling and Using YATDFT 

We use Intel Parallel Studio to compile Simint, Libxc, and YATDFT here. Simint and Libxc can also be compiled using GCC, but we have not tested yet. 

### 1. Compiling Simint

Notice: If possible, use ICC 17 instead of ICC 18 to compile Simint. It seems that ICC 18 will incorrectly optimize for some Simint functions. 

```shell
# Build Simint source code generator
# Note: not necessary to use ICC here
cd $WORKTOP
git clone https://github.com/gtfock-chem/simint-generator.git
cd simint-generator
mkdir build
cd build
CC=icc CXX=icpc cmake ../
make -j16
cd ..

# Generate Simint source code
# If your system does not use python 3 as default python interpretor, you can also use python2 to run the generating script
# Run ./create.py --help to see the details of the parameters
./create.py -g build/generator/ostei -l 3 -p 3 -d 0 -ve 4 -vg 5 -he 4 -hg 5 simint
mv simint ../

# Compile Simint
cd ../simint
# See the README file in Simint directory before choosing SIMINT_VECTOR. 
# For Skylake or later Xeon processors, you can use -DSIMINT_VECTOR=micavx512 
# and replace "xMIC-AVX512" in build-avx512/simint/CMakeFiles/simint.dir/flags.make 
# with "xCORE-AVX512".
# Don't set SIMINT_C_FLAGS if you do not need to profile or debug.
mkdir build-avx512   
CC=icc CXX=icpc cmake ../ -DSIMINT_VECTOR=micavx512 -DSIMINT_C_FLAGS="-O3;-g" -DCMAKE_INSTALL_PREFIX=./install
make -j16 install
```

### 2. Compiling Libxc

We compile Libxc into a shared library here since its static library is too large...

```shell
cd $WORKTOP
git clone https://gitlab.com/libxc/libxc.git
cd libxc
autoreconf -i
# Install the compiled Libxc in $WORKTOP/libxc/install
CC=icc CXX=icpc FC=ifort ./configure --prefix=$PWD/install --enable-shared
make -j16 
make check
make install
export LD_LIBRARY_PATH=$PWD/install/lib:$LD_LIBRARY_PATH
```

### 3. Compiling YATDFT library and Demo Programs

Modify `YATDFT/src/Makefile` and `YATDFT/tests/Makefile` according to the path of your compiled Simint and Libxc. Then just run `make` in these two directories to compile YATDFT library and demo programs. 

### Note for libCMS module

libCMS is a simplified version of the [libcint](https://github.com/gtfock-chem/libcint) library used by [GTFock](https://github.com/gtfock-chem/gtfock). We don't want to use the name "libcint" since it is the name of another ERI library since 2016. libCMS is responsible for:

* parsing input molecule coordinate file, basis set file and SAD initial guess file;
* providing data structure and functions for storing and accessing the information of the input chemical system;
* providing data structure for storing shell quartet information required by Simint and functions for computing batched/non-batching ERIs using Simint.

libCMS can also be used in other programs. To use libCMS, `libCMS.h` is the header file that should be included by external programs. 


Notice:

* Input coordinates are in Angstroms;
* The second (comment) line in the xyz file contains the net charge of the system;
* The basis set type should be Cartesian;
* The SCF initial guess for the density matrix is constructed from the superposition of atomic densities (SAD).  The densities for a few atoms are read automatically from the directory containing the basis set file.  These files are assumed be compatible with the basis set being used.

