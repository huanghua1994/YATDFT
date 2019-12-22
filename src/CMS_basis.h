#ifndef __CMS_BASISSET_H__
#define __CMS_BASISSET_H__

#include "CMS_config.h"

struct BasisSet
{
    // molecular information from xyz file
    int natoms;            // number of atoms in molecule
    int *eid;              // atomic numbers
    double *xn;            // x coords
    double *yn;            // y coords
    double *zn;            // z coords
    double *charge;        // double precision version of atomic numbers
    int nelectrons;        // sum of atomic numbers in molecule (not really num electrons)
    double **guess;        // initial guesses for each element in basis set (should not be in this section)
    int Q;                 // net charge read from xyz file (not related to nelectrons)
    double ene_nuc;        // nuclear energy (computed)

    // basis set information from gbs file
    int bs_nelements;      // max number of elements supported in basis set
    int bs_natoms;         // number of elements in basis set
    int basistype;         // Cartesian or spherical
    int *bs_eid;           // atomic numbers of elements in basis set
    int *bs_eptr;          // map atomic number to entry in basis set (array of len bs_nelements)
    int *bs_atom_start;    // start of element data in arrays of length nshells (array of length natoms+1)
    int bs_nshells;        // number of shells in the basis set (not the molecule)
    int bs_totnexp;        // total number of primitive functions in basis set
    int *bs_nexp;          // number of primitive functions for shell
    double **bs_exp;       // bs_exp[i] = orbital exponents for shell i
    double **bs_cc;        // bs_cc[i]  = contraction coefs for shell i
    double **bs_norm;      // bs_norm[i] = normalization constants for shell i
    int *bs_momentum;      // bs_momentum[i] = angular momentum for shell i
    
    // shell information for each shell in the given molecule
    uint32_t nshells;      // number of shells in given molecule
    uint32_t nfunctions;   // number of basis functions for molecule
    uint32_t *f_start_id;  // offset for first basis function for each shell
    uint32_t *f_end_id;    // offset for last basis function for each shell
    uint32_t *s_start_id;  // start of shell info for each atom
    uint32_t *nexp;        // number of primitives for each shell
    double **exp;          // exponents for each shell in molecule
    double *minexp;        // ?
    double **cc;           // contraction coefficients for each shell in molecule
    double **norm;         // ?
    uint32_t *momentum;    // angular momentum for each shell in molecule
    double *xyz0;          // centers for each shell in molecule, stored as linear array

    uint32_t maxdim;       // max number of functions among all shells in molecule
    uint32_t max_momentum;
    uint32_t max_nexp;
    uint32_t max_nexp_id;
    
    int mem_size;
    
    char str_buf[512];
};

typedef struct BasisSet *BasisSet_t;

#ifdef __cplusplus
extern "C" {
#endif

CMSStatus_t CMS_createBasisSet(BasisSet_t *basis);

CMSStatus_t CMS_destroyBasisSet(BasisSet_t basis);

CMSStatus_t CMS_loadChemicalSystem(BasisSet_t basis, char *bsfile, char *xyzfile );

int CMS_getNumAtoms(BasisSet_t basis);

int CMS_getNumShells(BasisSet_t basis);

int CMS_getNumFuncs(BasisSet_t basis);

int CMS_getNumOccOrb(BasisSet_t basis);

int CMS_getFuncStartInd(BasisSet_t basis, int shellid);

int CMS_getFuncEndInd(BasisSet_t basis, int shellid);

int CMS_getShellDim(BasisSet_t basis, int shellid);

int CMS_getMaxShellDim(BasisSet_t basis);

int CMS_getAtomStartInd(BasisSet_t basis, int atomid);

int CMS_getAtomEndInd(BasisSet_t basis, int atomid);

int CMS_getTotalCharge(BasisSet_t basis);

int CMS_getNneutral(BasisSet_t basis);

int CMS_getMaxMomentum(BasisSet_t basis);

int CMS_getMaxPrimid(BasisSet_t basis);

int CMS_getMaxnumExp(BasisSet_t basis);

double CMS_getNucEnergy(BasisSet_t basis);

void CMS_getInitialGuess(BasisSet_t basis, int atomid, double **guess, int *spos, int *epos);

void CMS_getShellxyz(BasisSet_t basis, int shellid, double *x, double *y, double *z);

// The following 4 functions are called by CMS_loadBasisSet, be careful and make sure you 
// understand what you are doing if you want to call any of them from external program
CMSStatus_t CMS_import_molecule(char *file, BasisSet_t basis);
CMSStatus_t CMS_import_basis(char *file, BasisSet_t basis);
CMSStatus_t CMS_parse_molecule(BasisSet_t basis);
CMSStatus_t CMS_import_guess(char *file, BasisSet_t basis);

#ifdef __cplusplus
}
#endif

#endif
