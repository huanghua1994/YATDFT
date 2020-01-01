#ifndef __CMS_SIMINT_H__
#define __CMS_SIMINT_H__

#include "simint/simint.h"

struct Simint
{
    int    nthread;
    int    max_am;
    int    workmem_per_thread;
    int    outmem_per_thread;
    double shell_memsize;
    double shellpair_memsize;
    double *workbuf;
    double *outbuf;

    int    nshells;
    struct simint_shell *shells;
    struct simint_multi_shellpair *shellpairs;
    
    int    df_nshells;
    int    df_max_am;
    int    *df_am_shell_id;
    int    *df_am_shell_spos;
    int    *df_am_shell_num;
    struct simint_shell *df_shells;
    struct simint_multi_shellpair *df_shellpairs;
    
    int    screen_method;
    double screen_tol;

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
#define _Simint_OSTEI_MAXAM 4
#define _Simint_NSHELL_SIMD 16

#define _Simint_AM_PAIRS (((_Simint_OSTEI_MAXAM) + 1) * ((_Simint_OSTEI_MAXAM) + 1))

#ifdef __cplusplus
extern "C" {
#endif

void CMS_Simint_init(BasisSet_t basis, Simint_t *simint, int nthread, double prim_scrval);

void CMS_Simint_setup_DF(Simint_t simint, BasisSet_t df_basis);

void CMS_Simint_free_DF_shellpairs(Simint_t simint);

void CMS_Simint_destroy(Simint_t simint, int show_stat);

int  CMS_Simint_get_sp_AM_idx(Simint_t simint, int P, int Q);

double CMS_Simint_get_DF_sp_scrval(Simint_t simint, int i);

void CMS_Simint_create_multi_sp(void **multi_sp_);

void CMS_Simint_free_multi_sp(void **multi_sp_);

void CMS_Simint_calc_pair_Hcore(
    BasisSet_t basis, Simint_t simint, int tid,
    int A, int B, double **integrals, int *nint
);

void CMS_Simint_calc_pair_ovlp(
    Simint_t simint, int tid, int A, int B, 
    double **integrals, int *nint
);

void CMS_Simint_calc_shellquartet(
    Simint_t simint, int tid, int M, int N, 
    int P, int Q, double **ERI, int *nint
);

// Computed batched 4-center integrals (M,N|P_list[i],Q_list[i])
void CMS_Simint_calc_shellquartet_batch(
    Simint_t simint, int tid, int M, int N, int *P_list, int *Q_list,
    int npair, double **batch_ERI, int *batch_nint, void **multi_sp_
);

// Compute density fitting 2-center integrals 
void CMS_Simint_calc_DF_shellpair(
    Simint_t simint, int tid, int M, int N,
    double **integrals, int *nint
);

// Compute batched density fitting 3-center integrals (M,N|P_i)
void CMS_Simint_calc_DF_shellquartet_batch(
    Simint_t simint, int tid, int M, int N, int *P_list, int npairs, 
    double **batch_ERI, int *batch_nint, void **multi_sp_
);

void CMS_Simint_add_accF_timer(Simint_t simint, double sec);

void CMS_Simint_reset_stat_info(Simint_t simint);

#ifdef __cplusplus
}
#endif

#endif