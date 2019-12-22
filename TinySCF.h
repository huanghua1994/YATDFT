#ifndef _YATSCF_TFOCK_H_
#define _YATSCF_TFOCK_H_

#include <omp.h>
#include "libCMS.h"

#define MAX_DIIS 10
#define MIN_DIIS 2

// Tiny SCF engine
struct TinySCF_struct 
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
    int    *valid_sp_lid;   // Left shell id of all screened unique shell pairs
    int    *valid_sp_rid;   // Right shell id of all screened unique shell pairs
    int    *shell_bf_sind;  // Index of the first basis function of each shell
    int    *shell_bf_num;   // Number of basis function in each shell
    double prim_scrtol;     // Primitive screening 
    double shell_scrtol2;   // Square of the shell screening tolerance
    double max_scrval;      // == max(fabs(sp_scrval(:)))
    double *sp_scrval;      // Square of screening values (upper bound) of each shell pair
    Simint_t   simint;      // Simint object for ERI, handled by libCMS
    BasisSet_t basis;       // Basis set object for storing chemical system info, handled by libCMS
    
    // Matrices and arrays used only in build_HF_mat
    int    max_JKacc_buf;   // Maximum buffer size for each thread's accumulating J and K matrix
    int    *blk_mat_ptr;    // Index of a given block's top-left element in the blocked matrix
    int    *Mpair_flag;     // Flags for marking if (M, i) is updated 
    int    *Npair_flag;     // Flags for marking if (N, i) is updated 
    double *J_blk_mat;      // Blocked J matrix
    double *K_blk_mat;      // Blocked K matrix
    double *D_blk_mat;      // Blocked D matrix
    double *JKacc_buf;      // All thread's buffer for accumulating J and K matrices
    double *FM_strip_buf;   // Thread-private buffer for F_MP and F_MQ blocks with the same M
    double *FN_strip_buf;   // Thread-private buffer for F_NP and F_NQ blocks with the same N

    // Temporary matrices used in multiple modules
    double *tmp_mat;        // build_Dmat, DIIS

    // Matrices and arrays used only in build_Dmat
    int    *ev_idx;         // Index of eigenvalues, for sorting
    double *eigval;         // Eigenvalues for building density matrix

    // Matrices and arrays used only in DIIS
    int    DIIS_len;        // Number of previous F matrices
    int    DIIS_bmax_id;    // The ID of a previous F matrix whose residual has the largest 2-norm
    int    *DIIS_ipiv;      // Permutation info for DGESV in DIIS
    double DIIS_bmax;       // The largest 2-norm of the stored F matrices' residuals
    double *F0_mat;         // Previous X^T * F * X matrices
    double *R_mat;          // "Residual" matrix
    double *B_mat;          // Linear system coefficient matrix in DIIS
    double *FDS_mat;        // F * D * S matrix in Commutator DIIS
    double *DIIS_rhs;       // Linear system right-hand-side vector in DIIS
    

    // SCF iteration info
    int    max_iter;        // Maximum SCF iteration
    int    iter;            // Current SCF iteration
    double nuc_energy;      // Nuclear energy
    double HF_energy;       // Hartree-Fock energy
    double ene_tol;         // SCF termination criteria for energy change
    
    // Matrices and temporary arrays in SCF
    double *Hcore_mat;      // Core Hamiltonian matrix
    double *S_mat;          // Overlap matrix
    double *F_mat;          // Fock matrix
    double *D_mat;          // Density matrix
    double *J_mat;          // Coulomb matrix
    double *K_mat;          // Exchange matrix
    double *X_mat;          // Basis transformation matrix
    double *Cocc_mat;       // Temporary matrix for building density matrix
    
    // Statistic 
    double mem_size, init_time, S_Hcore_time, shell_scr_time;
};

typedef struct TinySCF_struct* TinySCF_t;

// Initialize TinySCF with a Cartesian basis set file (.gbs format), a molecule 
// coordinate file and the number of SCF iterations (handled by libcint), and
// allocate all memory for other calculation
void TinySCF_init(TinySCF_t TinySCF, char *bas_fname, char *xyz_fname, const int max_iter);

// Compute the screening values of each shell quartet and the unique shell pairs
// that survive screening using Schwarz inequality
void TinySCF_compute_sq_Schwarz_scrvals(TinySCF_t TinySCF);

// Generate initial guess for density matrix using SAD data (handled by libcint)
void TinySCF_get_initial_guess(TinySCF_t TinySCF);

// Perform SCF iterations
void TinySCF_do_SCF(TinySCF_t TinySCF);

// Destroy TinySCF, free all allocated memory
void TinySCF_destroy(TinySCF_t TinySCF);

#endif
