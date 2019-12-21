## libCMS: C Middleware for Simint

libCMS is a simplified version of the [libcint](https://github.com/gtfock-chem/libcint) library used by [GTFock](https://github.com/gtfock-chem/gtfock). I don't want to use the name "libcint" since it is the name of a new (after 2016) ERI library. libCMS is responsible for:

1.  Parsing input molecule coordinate file, basis set file and SAD initial guess file;
2.  Providing data structure and functions for storing and accessing the information of the input chemical system;
3.  Providing data structure for storing shell quartet information required by Simint and functions for computing batched/non-batching ERIs using Simint.

libCMS can only be used on CPU. To use libCMS, `CMS.h` is the header file that should be included by external programs. 



Notice:

* Input coordinates are in Angstroms;
* The second (comment) line in the xyz file contains the net charge of the system;
* The basis set type should be Cartesian;
* The SCF initial guess for the density matrix is constructed from the superposition of atomic densities (SAD).  The densities for a few atoms are read automatically from the directory containing the basis set file.  These files are assumed be compatible with the basis set being used.