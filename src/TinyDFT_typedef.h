#ifndef _YATDFT_TYPEDEF_H_
#define _YATDFT_TYPEDEF_H_

#include <xc.h>
#include "libCMS.h"

// TinyDFT structure
struct TinyDFT_struct 
{
    int    nthread;         // Number of threads
    
    // Molecular system and ERI info
    char   *bas_name;       // Basis set file name
    char   *mol_name;       // Molecular file name
    int    natom;           // Number of atoms
    int    nshell;          // Number of shells
    int    nbf;             // Number of basis functions
    int    n_occ;           // Number of occupied orbits
    int    charge;          // Charge of molecule
    int    electron;        // Number of electrons
    int    num_total_sp;    // Number of total shell pairs (== nshell * nshell)
    int    num_valid_sp;    // Number of unique screened shell pairs
    int    mat_size;        // Size of matrices  (== nbf * nbf)
    int    max_dim;         // Maximum value of dim{M, N, P, Q}
    int    *valid_sp_lid;   // Size num_valid_sp, left shell id of all screened unique shell pairs
    int    *valid_sp_rid;   // Size num_valid_sp, right shell id of all screened unique shell pairs
    int    *shell_bf_sind;  // Size nshell+1, index of the first basis function of each shell
    int    *shell_bf_num;   // Size nshell, Number of basis function in each shell
    double prim_scrtol;     // Primitive screening 
    double shell_scrtol2;   // Square of the shell screening tolerance
    double *sp_scrval;      // Size num_total_sp, square of screening values (upper bound) of each shell pair
    Simint_t   simint;      // Simint object for ERI, handled by libCMS
    BasisSet_t basis;       // Basis set object for storing chemical system info, handled by libCMS
    
    // Flattened Gaussian basis function and atom info used only in XC calculation
    int    max_nprim;       // Maximum number of primitive functions in all shells
    int    *atom_idx;       // Size natom, atom index (H:1, He:2, Li:3, ...)
    int    *bf_nprim;       // Size nbf, number of primitive functions in each basis function
    double *atom_xyz;       // Size 3-by-natom, each column is an atom coordinate 
    double *bf_coef;        // Size nbf-by-max_nprim, coef  terms of basis functions
    double *bf_alpha;       // Size nbf-by-max_nprim, alpha terms of basis functions
    double *bf_exp;         // Size nbf-by-3, polynomial exponents terms of basis functions
    double *bf_center;      // Size nbf-by-3, center of basis functions
    
    // Matrices and arrays used only in build_HF_mat
    int    max_JKacc_buf;   // Maximum buffer size for each thread's accumulating J and K matrix
    int    *blk_mat_ptr;    // Size num_total_sp, index of a given block's top-left element in the blocked matrix
    int    *Mpair_flag;     // Size nshell*nthread, flags for marking if (M, i) is updated 
    int    *Npair_flag;     // Size nshell*nthread, flags for marking if (N, i) is updated 
    double *J_blk_mat;      // Size nbf-by-nbf, blocked J matrix
    double *K_blk_mat;      // Size nbf-by-nbf, blocked K matrix
    double *D_blk_mat;      // Size nbf-by-nbf, blocked D matrix
    double *JKacc_buf;      // Size unknown, all thread's buffer for accumulating J and K matrices
    double *FM_strip_buf;   // Size unknown, thread-private buffer for F_MP and F_MQ blocks with the same M
    double *FN_strip_buf;   // Size unknown, thread-private buffer for F_NP and F_NQ blocks with the same N

    // Matrices and arrays used in XC functional calculation
    int    xf_id;           // Exchange functional ID, default is LDA_X
    int    cf_id;           // Correlation functional ID, default is LDA_C_XALPHA
    int    xf_impl;         // If we has built-in implementation of the exchange functional
    int    cf_impl;         // If we has built-in implementation of the correlation functional
    int    xf_family;       // Exchange functional family (LDA / GGA)
    int    cf_family;       // Correlation functional family (LDA / GGA)
    int    nintp;           // Total number of XC numerical integral points
    int    nintp_blk;       // Maximum number of XC numerical integral points per block
    double *int_grid;       // Size 4-by-nintp, integral points and weights
    double *phi;            // Size nbf-by-nintp_blk, phi[i, :] are i-th basis 
                            // function values at some integral points
    double *rho;            // Size nintp_blk, electron density at some integral points
    double *exc;            // Size nintp_blk, "energy per unit particle"
    double *vxc;            // Size nintp_blk, correlation potential
    double *XC_workbuf;     // Size nbf*nintp_blk, XC calculation work buffer
    xc_func_type libxc_xf;  // Libxc exchange functional handle
    xc_func_type libxc_cf;  // Libxc correlation functional handle

    // Temporary matrices used in multiple modules
    double *tmp_mat;        // Size nbf-by-nbf, used in: build_Dmat, CDIIS

    // Matrices and arrays used only in build_Dmat
    int    *ev_idx;         // Size nbf, index of eigenvalues, for sorting
    double *eigval;         // Size nbf, eigenvalues for building density matrix

    // Matrices and arrays used only in CDIIS
    int    DIIS_len;        // Number of previous F matrices
    int    DIIS_bmax_id;    // The ID of a previous F matrix whose residual has the largest 2-norm
    int    *DIIS_ipiv;      // Size MAX_DIIS+1, permutation info for DGESV in DIIS
    double DIIS_bmax;       // The largest 2-norm of the stored F matrices' residuals
    double *F0_mat;         // Size nbf*nbf*MAX_DIIS, previous X^T * F * X matrices
    double *R_mat;          // Size nbf-by-nbf, "residual" matrix
    double *B_mat;          // Size (MAX_DIIS+1)^2, linear system coefficient matrix in DIIS
    double *FDS_mat;        // Size nbf-by-nbf, F * D * S matrix in Commutator DIIS
    double *DIIS_rhs;       // Size MAX_DIIS+1, Linear system right-hand-side vector in DIIS

    // Matrices and arrays used in SCF
    // Except Cocc_mat is nbf-by-n_occ, all other matrices are nbf-by-nbf
    double *Hcore_mat;      // Core Hamiltonian matrix
    double *S_mat;          // Overlap matrix
    double *X_mat;          // Basis transformation matrix
    double *J_mat;          // Coulomb matrix
    double *K_mat;          // Hartree-Fock exchange matrix
    double *XC_mat;         // Exchange-correlation functional matrix
    double *F_mat;          // Fock matrix (from Hcore, J, K/XC)
    double *D_mat;          // Density matrix
    double *Cocc_mat;       // Factor of density matrix

    // Calculated energies
    double E_nuc_rep;       // Nuclear repulsion energy
    double E_one_elec;      // One-electron integral energy, == sum(sum(2 .* D .* Hcore))
    double E_two_elec;      // Two-electron integral energy, == sum(sum(2 .* D .* J))
    double E_HF_exchange;   // Hartree-Fock exchange energy, == sum(sum(-D .* K))
    double E_DFT_XC;        // DFT exchange-correlation energy

    // SCF iteration info
    int    max_iter;        // Maximum SCF iteration
    int    iter;            // Current SCF iteration
    double E_tol;           // SCF termination criteria for energy change
    
    // Statistic 
    double mem_size, init_time, S_Hcore_time, shell_scr_time;
};

typedef struct TinyDFT_struct* TinyDFT_t;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize a TinyDFT structure, including:
//   (1) load molecular system and preparing ERI related data structures using libCMS;
//   (2) allocate memory for all matrices;
//   (3) perform Schwarz screening for shell pairs.
// Input parameters:
//   bas_fname : Gaussian basis set file name (.gbs)
//   xyz_fname : Molecule coordinate file name
// Output parameter:
//   TinyDFT_ : Pointer to a initialized TinyDFT structure
void TinyDFT_init(TinyDFT_t *TinyDFT_, char *bas_fname, char *xyz_fname);

// Destroy a TinyDFT structure
// Input parameter:
//   TinyDFT_ : Pointer to a TinyDFT structure to be destroyed
void TinyDFT_destroy(TinyDFT_t *_TinyDFT);

#ifdef __cplusplus
}
#endif

#endif
