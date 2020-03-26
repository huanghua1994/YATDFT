#ifndef __CMS_SIMINT_H__
#define __CMS_SIMINT_H__

#include "simint/simint.h"

struct Simint
{
    int    nthread;             // Number of threads that will call Simint
    int    nshell;              // Number of regular basis set shells
    int    max_am;              // Maximum AM number in regular basis set shells
    int    df_nshell;           // Number of density fitting (DF) basis set shells
    int    df_max_am;           // Maximum AM number in DF basis set shells
    int    workmem_per_thread;  // Size of Simint work memory (double) required for each thread
    int    outmem_per_thread;   // Size of Simint ERI output memory (double) required for each thread
    int    screen_method;       // Simint screening method
    int    *df_am_shell_id;     // Size df_nshell, original DF shell id of each sorted DF shells
    int    *df_am_shell_spos;   // Size df_max_am+2, index of first DF shell of each AM
    int    *df_am_shell_num;    // Size df_max_am+1, number of DF shells of each AM
    double screen_tol;          // Screening tolerance
    double shell_memsize;       // Total memory size of shells
    double shellpair_memsize;   // Total memory size of shell pairs
    double *workbuf;            // Size workmem_per_thread * nthread, Simint work memory
    double *outbuf;             // Size outmem_per_thread  * nthread, Simint ERI output memory

    // Shells and shell pairs of regular and DF basis sets
    struct simint_shell *shells;                   // Size nshell
    struct simint_shell *df_shells;                // Size df_nshell+1, the last one is unit shell
    struct simint_multi_shellpair *shellpairs;     // Size nshell * nshell
    struct simint_multi_shellpair *df_shellpairs;  // Size df_nshell

    // For timer, only master thread will write to these
    double ostei_actual, ostei_setup, fock_update_F;

    // For statistic
    double *num_multi_shellpairs, *sum_nprim;
    double *num_screened_prim, *num_unscreened_prim;
    double *num_screened_vec,  *num_unscreened_vec;
};

typedef struct Simint *Simint_t;

// The following 2 constants are corresponding to Simint_OSTEI_MAXAM
// and Simint_NSHELL_SIMD in Simint. I cannot include <simint/simint.h>
// here, so I just update the values manually. 
#define _SIMINT_OSTEI_MAXAM 4
#define _SIMINT_NSHELL_SIMD 16

#define _SIMINT_AM_PAIRS (((_SIMINT_OSTEI_MAXAM) + 1) * ((_SIMINT_OSTEI_MAXAM) + 1))

#ifdef __cplusplus
extern "C" {
#endif

// Initialize a Simint structure with a BasisSet structure
// Input parameters:
//   basis       : Initialized BasisSet structure for regular basis set
//   nthread     : Number of threads that will call Simint
//   prim_scrval : Primitive integral screening threshold (usually 1e-14)
// Output parameter:
//   simint : Initialized Simint structure
void CMS_Simint_init(BasisSet_t basis, Simint_t *simint, int nthread, double prim_scrval);

// Create Simint shell pair structures for unique screened shell pairs
// Input parameters:
//   nsp        : Number of unique screened shell pairs
//   {M,N}_list : Shell pair (M_list[i], N_list[i]) is a unique screened shell pair
// Output parameter: 
//   simint : Simint structure with initialized unique screened shell pairs
void CMS_Simint_create_uniq_scr_sp(Simint_t simint, const int nsp, const int *M_list, const int *N_list);

// Set up density fitting (DF) related data structures in a Simint structure
// Input parameters:
//   simint   : Initialized Simint structure
//   df_basis : Initialized BasisSet structure for DF basis set
// Output parameter:
//   simint : Simint structure with initialized DF related data structures
void CMS_Simint_setup_DF(Simint_t simint, BasisSet_t df_basis);

// Free shells and shell pairs used in DF after constructing the DF tensor.
// Not necessary, just to save some memory as soon as possible. If this function is 
// not called, those shells and shell pairs will be released in CMS_Simint_destroy().
// Input parameter:
//   simint : Simint structure with initialized DF related data structures
// Output parameter:
//   simint : Simint structure without initialized DF related data structures
void CMS_Simint_free_DF_shellpairs(Simint_t simint);

// Destroy a Simint structure and free all memory
// Input parameters:
//   simint    : Simint structure to be destroyed
//   show_stat : 0/1, 1 will show Simint statistic information
void CMS_Simint_destroy(Simint_t simint, int show_stat);

// Get the AM pair index of a shell pair
// Input parameters:
//   simint : Simint structure
//   P, Q   : Regular shell pair (P, Q) in simint
// Output parameter:
//   <return> : AM(P) * _SIMINT_OSTEI_MAXAM + AM(Q)
int  CMS_Simint_get_sp_AM_idx(Simint_t simint, int P, int Q);

// Get the shell pair screening value of the i-th DF shell pair
// Input parameters:
//   simint : Simint structure with initialized DF related data structures
//   i      : DF shell pair index
// Output parameter:
//   <return> : Shell pair screening value of the i-th DF shell pair
double CMS_Simint_get_DF_sp_scrval(Simint_t simint, int i);

// Create a simint_multi_shellpair structure and return the pointer to it
// Output parameter:
//   *multi_sp_ : Pointer to an initialized simint_multi_shellpair structure
void CMS_Simint_create_multi_sp(void **multi_sp_);

// Destroy a simint_multi_shellpair structure
// Input parameter:
//   multi_sp : Pointer to a simint_multi_shellpair structure to be destroyed
void CMS_Simint_free_multi_sp(void *multi_sp);

// Calculate a block of the core Hamiltonian (Hcore) matrix corresponding to a given shell pair
// Input parameters:
//   basis  : Initialized BasisSet structure
//   simint : Initialized Simint structure
//   tid    : Thread ID
//   A, B   : Shell pair indices
// Output parameters:
//   *integrals : Pointer to the calculated block, size NCART(AM(A)) * NCART(AM(B))
//   *nint      : 0 if simint is not initialized, otherwise == NCART(AM(A)) * NCART(AM(B))
void CMS_Simint_calc_pair_Hcore(
    BasisSet_t basis, Simint_t simint, int tid,
    int A, int B, double **integrals, int *nint
);

// Calculate a block of the overlapping (S) matrix corresponding to a given shell pair
// Input parameters:
//   simint : Initialized Simint structure
//   tid    : Thread ID
//   A, B   : Shell pair indices
// Output parameters:
//   *integrals : Pointer to the calculated block, size NCART(AM(A)) * NCART(AM(B))
//   *nint      : 0 if simint is not initialized, otherwise == NCART(AM(A)) * NCART(AM(B))
void CMS_Simint_calc_pair_ovlp(
    Simint_t simint, int tid, int A, int B, 
    double **integrals, int *nint
);

// Calculate a shell quartet (M,N|P,Q)
// Input parameters:
//   simint     : Initialized Simint structure
//   tid        : Thread ID
//   M, N, P, Q : Shell indices
// Output parameters:
//   *ERI  : Pointer to the calculated block, size 
//           NCART(AM(M)) * NCART(AM(N)) * NCART(AM(P)) * NCART(AM(Q))
//   *nint : 0 if simint is not initialized, otherwise == 
//           NCART(AM(M)) * NCART(AM(N)) * NCART(AM(P)) * NCART(AM(Q))
void CMS_Simint_calc_shellquartet(
    Simint_t simint, int tid, int M, int N, 
    int P, int Q, double **ERI, int *nint
);

// Calculate a shell quartet (M,N|M,N) for screening
// Input parameters:
//   simint     : Initialized Simint structure
//   tid        : Thread ID
//   M, N       : Shell indices
//   *multi_sp_ : Pointer to thread-private initialized simint_multi_shellpair structure
// Output parameters:
//   *ERI  : Pointer to the calculated block, size 
//           NCART(AM(M)) * NCART(AM(N)) * NCART(AM(M)) * NCART(AM(N))
//   *nint : 0 if simint is not initialized, otherwise == 
//           NCART(AM(M)) * NCART(AM(N)) * NCART(AM(M)) * NCART(AM(N))  
void CMS_Simint_calc_MNMN_shellquartet(
    Simint_t simint, int tid, int M, int N, 
    void **multi_sp_, double **ERI, int *nint
);

// Calculate a batch of shell quartets (M,N|P_list[i],Q_list[i])
// Input parameters:
//   simint     : Initialized Simint structure
//   tid        : Thread ID
//   M, N       : Bra-side shell indices
//   npair      : Number of ket-side shell pairs, should <= Simint_NSHELL_SIMD
//   P_list     : Size npair, list of P shell indices, all P shells must have the same AM
//   Q_list     : Size npair, list of Q shell indices, all Q shells must have the same AM
//   *multi_sp_ : Pointer to thread-private initialized simint_multi_shellpair structure
// Output parameters:
//   *batch_ERI  : Pointer to the calculated blocks, size of each block is 
//                 NCART(AM(M)) * NCART(AM(N)) * NCART(AM(P)) * NCART(AM(Q))
//   *batch_nint : 0 if simint is not initialized, otherwise == 
//                 NCART(AM(M)) * NCART(AM(N)) * NCART(AM(P)) * NCART(AM(Q))
void CMS_Simint_calc_shellquartet_batch(
    Simint_t simint, int tid, int M, int N, int npair, int *P_list, 
    int *Q_list, double **batch_ERI, int *batch_nint, void **multi_sp_
);

// Calculate a DF 2-center integral (M,I|N,I) where I is unit shell
// Input parameters:
//   simint : Initialized Simint structure
//   tid    : Thread ID
//   M, N   : DF shell indices
// Output parameters:
//   *integrals : Pointer to the calculated block, size NCART(AM(M)) * NCART(AM(N))
//   *nint      : 0 if simint is not initialized, otherwise == NCART(AM(M)) * NCART(AM(N))
void CMS_Simint_calc_DF_shellpair(
    Simint_t simint, int tid, int M, int N,
    double **integrals, int *nint
);

// Calculate a batch of DF 3-center integrals (M,N|P_list[i],I) where I is unit shell
// Input parameters:
//   simint     : Initialized Simint structure
//   tid        : Thread ID
//   M, N       : Bra-side regular shell indices
//   npair      : Number of ket-side shell pairs, should <= Simint_NSHELL_SIMD
//   P_list     : Size npair, list of P DF shell indices, all P shells must have the same AM
//   *multi_sp_ : Pointer to thread-private initialized simint_multi_shellpair structure
// Output parameters:
//   *batch_ERI  : Pointer to the calculated blocks, size of each block is 
//                 NCART(AM(M)) * NCART(AM(N)) * NCART(AM(P))
//   *batch_nint : 0 if simint is not initialized, otherwise == 
//                 NCART(AM(M)) * NCART(AM(N)) * NCART(AM(P))
void CMS_Simint_calc_DF_shellquartet_batch(
    Simint_t simint, int tid, int M, int N, int npair, int *P_list, 
    double **batch_ERI, int *batch_nint, void **multi_sp_
);

// Add Fock matrix accumulation time to Simint structure timer
// Input parameters:
//   simint : Simint structure
//   sec    : Fock matrix accumulation time, in seconds
void CMS_Simint_add_accF_timer(Simint_t simint, double sec);

// Reset all Simint statistic information
// Input parameter:
//   simint : Simint structure to be reset
void CMS_Simint_reset_stat_info(Simint_t simint);

#ifdef __cplusplus
}
#endif

#endif