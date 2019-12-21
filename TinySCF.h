#ifndef _YATSCF_TFOCK_H_
#define _YATSCF_TFOCK_H_

#include <omp.h>
#include "libCMS.h"

#define MAX_DIIS 10
#define MIN_DIIS 2

// Tiny SCF engine
struct TinySCF_struct 
{
    // OpenMP parallel setting and buffer
    int    nthreads;        // Number of threads
    int    max_buf_size;    // Maximum buffer size for each thread's accumulating Fock matrix
    double *Accum_Fock_buf; // Pointer to all thread's buffer for accumulating Fock matrix
    
    // Chemical system info
    BasisSet_t basis;     // Basis set object for storing chemical system info, handled by libCMS
    int natoms, nshells;  // Number of atoms and shells
    int nbasfuncs, n_occ; // Number of basis functions and occupied orbits
    int charge, electron; // Charge and number of electrons 
    char *bas_name, *mol_name;
    
    // Auxiliary variables 
    int nshellpairs;      // Number of shell pairs        (== nshells * nshells)
    int num_uniq_sp;      // Number of unique shell pairs (== nshells * (nshells+1) / 2)
    int mat_size;         // Size of matrices             (== nbasfuncs * nbasfuncs)
    int max_dim;          // Maximum value of dim{M, N, P, Q}
    
    // SCF iteration info
    int    niters, iter;  // Maximum and current SCF iteration
    double nuc_energy;    // Nuclear energy
    double HF_energy;     // Hartree-Fock energy
    double ene_tol;       // SCF termination criteria for energy change
    
    // Screening parameters
    double prim_scrtol;   // Primitive screening 
    double shell_scrtol2; // Square of the shell screening tolerance
    double max_scrval;    // == max(fabs(sp_scrval(:)))
    double *sp_scrval;    // Square of screening values (upper bound) of each shell pair
    int    *uniq_sp_lid;  // Left shell id of all unique shell pairs
    int    *uniq_sp_rid;  // Right shell id of all unique shell pairs
    
    // ERIs
    Simint_t simint;      // Simint object for ERI, handled by libCMS
    int *shell_bf_sind;   // Index of the first basis function of each shell
    int *shell_bf_num;    // Number of basis function in each shell
    
    // Matrices and temporary arrays in SCF
    double *Hcore_mat;    // Core Hamiltonian matrix
    double *S_mat;        // Overlap matrix
    double *F_mat;        // Fock matrix
    double *D_mat;        // Density matrix
    double *J_mat;        // Coulomb matrix   ((J+J^T)/2 is the actual Coulomb matrix)
    double *K_mat;        // Exchange matrix  ((K+K^T)/2 is the actual Exchange matrix) 
    double *X_mat;        // Basis transformation matrix
    double *Cocc_mat;     // Temporary matrix for building density matrix
    double *eigval;       // Eigenvalues for building density matrix
    int    *ev_idx;       // Index of eigenvalues, for sorting
    double *tmp_mat;      // Temporary matrix
    double *D2_mat;       // Temporary matrix used in purification
    double *D3_mat;       // Temporary matrix used in purification
    
    // Blocked J, K and D matrices and the offsets of each block
    double *J_mat_block;      // Blocked J matrix
    double *K_mat_block;      // Blocked K matrix
    double *D_mat_block;      // Blocked D matrix
    int    *mat_block_ptr;    // Index of a given block's top-left element in the blocked matrix
    double *F_M_band_blocks;  // Thread-private buffer for F_MP and F_MQ blocks with the same M
    double *F_N_band_blocks;  // Thread-private buffer for F_NP and F_NQ blocks with the same N
    int    *visited_Mpairs;   // Flags for marking if (M, i) is updated 
    int    *visited_Npairs;   // Flags for marking if (N, i) is updated 
    
    // Matrices and arrays for DIIS
    double *F0_mat;       // Previous X^T * F * X matrices
    double *R_mat;        // "Residual" matrix
    double *B_mat;        // Linear system coefficient matrix in DIIS
    double *FDS_mat;      // F * D * S matrix in Commutator DIIS
    double *DIIS_rhs;     // Linear system right-hand-side vector in DIIS
    int    *DIIS_ipiv;    // Permutation info for DGESV in DIIS
    int    DIIS_len;      // Number of previous F matrices
    int    DIIS_bmax_id;  // The ID of a previous F matrix whose residual has the largest 2-norm
    double DIIS_bmax;     // The largest 2-norm of the stored F matrices' residuals
    
    // Statistic 
    double mem_size, init_time, S_Hcore_time, shell_scr_time;
};

typedef struct TinySCF_struct* TinySCF_t;

// Initialize TinySCF with a Cartesian basis set file (.gbs format), a molecule 
// coordinate file and the number of SCF iterations (handled by libcint), and
// allocate all memory for other calculation
void TinySCF_init(TinySCF_t TinySCF, char *bas_fname, char *xyz_fname, const int niters);

// Compute core Hamiltonian and overlap matrix, and generate basis transform matrix
// The overlap matrix is not needed after generating basis transform matrix, and its
// memory space will be used as temporary space in other procedure
void TinySCF_compute_Hcore_Ovlp_mat(TinySCF_t TinySCF);

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
