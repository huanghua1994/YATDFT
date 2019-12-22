# YATDFT: Yet Another Tiny DFT

A tiny library for constructing matrices used in Hartree-Fock (HF) and Kohn-Sham density functional theory (KS-DFT) using Gaussian basis sets. 

YATDFT can construct:

* Core Hamiltonian matrix
* Overlap matrix
* Basis transformation matrix
* Coulomb matrix
* HF exchange matrix
* CDIIS Pulay mixing for Fock matrix
* Density matrix (from Fock matrix or SAD initial guess)
* To be added: DFT exchange-correlation matrix (LDA)

YATDFT requires:

* [Simint](https://github.com/simint-chem/simint-generator)
* OpenMP and C99 supported C compiler
* Intel MKL



### README for libCMS module

libCMS is a simplified version of the [libcint](https://github.com/gtfock-chem/libcint) library used by [GTFock](https://github.com/gtfock-chem/gtfock). I don't want to use the name "libcint" since it is the name of a new (after 2016) ERI library. libCMS is responsible for:

* Parsing input molecule coordinate file, basis set file and SAD initial guess file;
* Providing data structure and functions for storing and accessing the information of the input chemical system;
* Providing data structure for storing shell quartet information required by Simint and functions for computing batched/non-batching ERIs using Simint.

libCMS can also be used in other programs. To use libCMS, `libCMS.h` is the header file that should be included by external programs. 


Notice:

* Input coordinates are in Angstroms;
* The second (comment) line in the xyz file contains the net charge of the system;
* The basis set type should be Cartesian;
* The SCF initial guess for the density matrix is constructed from the superposition of atomic densities (SAD).  The densities for a few atoms are read automatically from the directory containing the basis set file.  These files are assumed be compatible with the basis set being used.

