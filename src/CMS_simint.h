#ifndef __CMS_SIMINT_H__
#define __CMS_SIMINT_H__

struct Simint
{
    int nthreads;
    int max_am;
    int workmem_per_thread;
    int outmem_per_thread;
    double shell_memsize;
    double shellpair_memsize;
    double *workbuf;
    double *outbuf;

    int nshells;
    struct simint_shell *shells;
    struct simint_multi_shellpair *shellpairs;

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

#ifdef __cplusplus
extern "C" {
#endif

CMSStatus_t CMS_createSimint(BasisSet_t basis, Simint_t *simint, int nthreads, double prim_scrval);

CMSStatus_t CMS_destroySimint(Simint_t simint, int show_stat);

CMSStatus_t
CMS_computeShellQuartet_Simint(Simint_t simint, int tid,
                                int A, int B, int C, int D,
                                double **integrals, int *nints);

CMSStatus_t
CMS_computePairOvl_Simint(BasisSet_t basis, Simint_t simint, int tid,
                           int A, int B,
                           double **integrals, int *nints);

CMSStatus_t
CMS_computePairCoreH_Simint(BasisSet_t basis, Simint_t simint, int tid,
                           int A, int B,
                           double **integrals, int *nints);


// The following 2 constants are corresponding to Simint_OSTEI_MAXAM
// and Simint_NSHELL_SIMD in Simint. I cannot include <simint/simint.h>
// here, so I just update the values manually. This problem should be 
// solved later.
#define _Simint_OSTEI_MAXAM 4
#define _Simint_NSHELL_SIMD 16

#define _Simint_AM_PAIRS (((_Simint_OSTEI_MAXAM) + 1) * ((_Simint_OSTEI_MAXAM) + 1))

void CMS_Simint_addupdateFtimer(Simint_t simint, double sec);

int  CMS_Simint_getShellpairAMIndex(Simint_t simint, int P, int Q);

void CMS_Simint_createThreadMultishellpair(void **thread_multi_shellpair);

void CMS_Simint_freeThreadMultishellpair(void **thread_multi_shellpair);

CMSStatus_t 
CMS_computeShellQuartetBatch_Simint(
    Simint_t simint, int tid,
    int M, int N, int *P_list, int *Q_list,
    int npair, double **thread_batch_integrals, int *thread_batch_nints,
    void **thread_multi_shellpairs
);

void CMS_Simint_resetStatisInfo(Simint_t simint);

#ifdef __cplusplus
}
#endif

#endif