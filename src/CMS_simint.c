#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include "simint/simint.h"
#include "CMS_config.h"
#include "CMS_basis.h"
#include "CMS_simint.h"

#define NCART(am) (((am)+1)*((am)+2)/2)

typedef struct simint_shell  shell_s;
typedef struct simint_shell* shell_t;
typedef struct simint_multi_shellpair  multi_sp_s;
typedef struct simint_multi_shellpair* multi_sp_t;

static double CMS_get_walltime_sec()
{
    double sec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}

void CMS_Simint_init(BasisSet_t basis, Simint_t *simint, int nthread, double prim_scrval)
{
    CMS_ASSERT(nthread > 0);

    Simint_t s = (Simint_t) calloc(1, sizeof(struct Simint));
    CMS_ASSERT(s != NULL);

    simint_init();

    s->nthread = nthread;
    s->max_am = basis->max_momentum;
    int max_ncart = NCART(s->max_am), buff_size;

    // Allocate workbuf for all threads on this node
    buff_size = simint_ostei_workmem(0, s->max_am);
    if (buff_size < max_ncart * max_ncart) buff_size = max_ncart * max_ncart;
    buff_size = (buff_size + 7) / 8 * 8;  // Align to 8 double (64 bytes)
    s->workmem_per_thread = buff_size;
    s->workbuf = (double *) _mm_malloc(s->workmem_per_thread * nthread * sizeof(double), 64);
    CMS_ASSERT(s->workbuf != NULL);

    // Allocate outbuf for all threads on this node
    // Output buffer should holds Simint_NSHELL_SIMD ERI results
    // +8 for Simint primitive screening statistic info 
    buff_size = max_ncart * max_ncart * max_ncart * max_ncart;
    buff_size = (buff_size + 7) / 8 * 8;   // Align to 8 double (64 bytes)
    s->outmem_per_thread = buff_size * _SIMINT_NSHELL_SIMD + 8;  
    s->outbuf = (double *) _mm_malloc(s->outmem_per_thread * nthread * sizeof(double), 64);
    CMS_ASSERT(s->outbuf != NULL);

    // Form and store Simint shells for all shells of this molecule
    int nshell = basis->nshells;
    size_t shells_msize = sizeof(shell_s) * nshell;
    s->nshell = nshell;
    s->shells = (shell_t) malloc(shells_msize);
    CMS_ASSERT(s->shells != NULL);
    s->shell_memsize = (double) shells_msize;

    shell_t shell_ptr = s->shells;
    for (int i=0; i < nshell; i++)
    {
        // Initialize variables in structure
        simint_initialize_shell(shell_ptr); 

        // Allocate space for alpha and coef for the shell
        simint_allocate_shell(basis->nexp[i], shell_ptr);
        s->shell_memsize += (double) shell_ptr->memsize;

        shell_ptr->am    = basis->momentum[i];
        shell_ptr->nprim = basis->nexp[i];
        shell_ptr->x     = basis->xyz0[i*4+0];
        shell_ptr->y     = basis->xyz0[i*4+1];
        shell_ptr->z     = basis->xyz0[i*4+2];

        for (int j=0; j<basis->nexp[i]; j++)
        {
            shell_ptr->alpha[j] = basis->exp[i][j];
            shell_ptr->coef[j]  = basis->cc[i][j];
        }

        shell_ptr++;
    }

    // Here we assume there are no unit shells (shells with zero orbital exponent)
    simint_normalize_shells(nshell, s->shells);

    // For primitive screening, fast Schwarz might have issue with aug-cc-pVDZ,
    // try to use SIMINT_SCREEN_SCHWARZ if necessary
    if (prim_scrval < 0.0 || prim_scrval > 1) prim_scrval = 1e-14;
    s->screen_method = SIMINT_SCREEN_SCHWARZ;
    s->screen_tol    = prim_scrval;
    printf("Simint screen method    = SIMINT_SCREEN_SCHWARZ \n");
    printf("Simint prim screen tol  = %.2e\n", s->screen_tol);

    // Precompute all shell pairs
    // Will be used by CMS_Simint_fill_multi_sp_list(), DO NOT SKIP it!!!
    double sp_msize = sizeof(multi_sp_s) * nshell * nshell;
    s->shellpairs = (multi_sp_t) malloc(sp_msize);
    CMS_ASSERT(s->shellpairs != NULL);
    s->shellpair_memsize = (double) sp_msize;

    // UNDONE: exploit symmetry
    for (int i = 0; i < nshell; i++)
    {
        for (int j = 0; j< nshell; j++)
        {
            multi_sp_t pair;
            pair = &s->shellpairs[i * nshell + j];
            simint_initialize_multi_shellpair(pair);
            simint_create_multi_shellpair(1, s->shells+i, 1, s->shells+j, pair, s->screen_method);
            s->shellpair_memsize += (double) pair->memsize;
        }
    }
    
    // Reset timer
    s->ostei_setup   = 0.0;
    s->ostei_actual  = 0.0;
    s->fock_update_F = 0.0;

    // Allocate space for statistic info
    int stat_info_size = sizeof(double) * nthread;
    s->num_multi_shellpairs = (double*) malloc(stat_info_size);
    s->sum_nprim            = (double*) malloc(stat_info_size);
    s->num_screened_prim    = (double*) malloc(stat_info_size);
    s->num_unscreened_prim  = (double*) malloc(stat_info_size);
    s->num_screened_vec     = (double*) malloc(stat_info_size);
    s->num_unscreened_vec   = (double*) malloc(stat_info_size);
    CMS_ASSERT(s->num_multi_shellpairs != NULL && s->sum_nprim           != NULL);
    CMS_ASSERT(s->num_screened_prim    != NULL && s->num_unscreened_prim != NULL);
    CMS_ASSERT(s->num_screened_vec     != NULL && s->num_unscreened_vec  != NULL);
    memset(s->num_multi_shellpairs, 0, stat_info_size);
    memset(s->sum_nprim,            0, stat_info_size);
    memset(s->num_screened_prim,    0, stat_info_size);
    memset(s->num_unscreened_prim,  0, stat_info_size);
    memset(s->num_screened_vec,     0, stat_info_size);
    memset(s->num_unscreened_vec,   0, stat_info_size);
    
    double workmem_MB = s->workmem_per_thread * 64 * sizeof(double) / 1048576.0;
    double outmem_MB  = s->outmem_per_thread  * 64 * sizeof(double) / 1048576.0;
    double shellpair_mem_MB = s->shellpair_memsize / 1048576.0;
    double stat_info_mem_MB = stat_info_size * 6   / 1048576.0;
    double Simint_mem_MB = workmem_MB + outmem_MB + outmem_MB + shellpair_mem_MB + stat_info_mem_MB;
    printf("CMS Simint memory usage = %.2lf MB \n", Simint_mem_MB);
    
    s->df_am_shell_id   = NULL;
    s->df_am_shell_spos = NULL;
    s->df_am_shell_num  = NULL;
    s->df_shells        = NULL;
    s->df_shellpairs    = NULL;
    
    *simint = s;
}

void CMS_Simint_setup_DF(Simint_t simint, BasisSet_t df_basis)
{
    Simint_t s = simint;
    
    // Reallocate workbuf for density fitting
    s->df_max_am = df_basis->max_momentum;
    if (s->df_max_am > s->max_am) s->max_am = s->df_max_am; 
    int max_ncart = NCART(s->max_am);
    int buff_size = simint_ostei_workmem(0, s->max_am);
    if (buff_size < max_ncart * max_ncart) buff_size = max_ncart * max_ncart;
    buff_size = (buff_size + 7) / 8 * 8; // Align to 8 double (64 bytes)
    s->workmem_per_thread = buff_size;
    _mm_free(s->workbuf);
    s->workbuf = (double *) _mm_malloc(s->workmem_per_thread * s->nthread * sizeof(double), 64);
    CMS_ASSERT(s->workbuf != NULL);
    
    // Form and store Simint shells for all density fitting shells
    // The last shell is the unit shell
    int df_nshell = df_basis->nshells;
    size_t df_shells_msize = sizeof(shell_s) * (df_nshell + 1);
    s->df_nshell = df_nshell;
    s->df_shells = (shell_t) malloc(df_shells_msize);
    CMS_ASSERT(s->shells != NULL);
    s->shell_memsize = (double) df_shells_msize;
    
    // Copy all density fitting shells 
    shell_t df_shell_ptr = s->df_shells;
    for (int i = 0; i < df_nshell; i++)
    {
        // Initialize variables in structure
        simint_initialize_shell(df_shell_ptr); 

        // Allocate space for alpha and coef for the shell
        simint_allocate_shell(df_basis->nexp[i], df_shell_ptr);
        s->shell_memsize += (double) df_shell_ptr->memsize;

        df_shell_ptr->am    = df_basis->momentum[i];
        df_shell_ptr->nprim = df_basis->nexp[i];
        df_shell_ptr->x     = df_basis->xyz0[i*4+0];
        df_shell_ptr->y     = df_basis->xyz0[i*4+1];
        df_shell_ptr->z     = df_basis->xyz0[i*4+2];

        for (int j = 0; j < df_basis->nexp[i]; j++)
        {
            df_shell_ptr->alpha[j] = df_basis->exp[i][j];
            df_shell_ptr->coef[j]  = df_basis->cc[i][j];
        }

        df_shell_ptr++;
    }
    // The unit shell
    simint_initialize_shell(df_shell_ptr); 
    simint_allocate_shell(1, df_shell_ptr);
    s->shell_memsize    += (double) df_shell_ptr->memsize;
    df_shell_ptr->am       = 0;
    df_shell_ptr->nprim    = 1;
    df_shell_ptr->x        = 0;
    df_shell_ptr->y        = 0;
    df_shell_ptr->z        = 0;
    df_shell_ptr->alpha[0] = 0;
    df_shell_ptr->coef[0]  = 1;
    
    // Normalize shells except the unit shells
    simint_normalize_shells(df_nshell, s->df_shells);
    
    // Precompute all shell pairs for density fitting, DO NOT SKIP IT
    size_t df_sp_msize = sizeof(multi_sp_s) * df_nshell;
    s->df_shellpairs = (multi_sp_t) malloc(df_sp_msize);
    CMS_ASSERT(s->df_shellpairs != NULL);
    s->shellpair_memsize += (double) df_sp_msize;
    int unit_shell_id = df_nshell;
    for (int i = 0; i < df_nshell; i++)
    {
        multi_sp_t pair;
        pair = &s->df_shellpairs[i];
        simint_initialize_multi_shellpair(pair);
        simint_create_multi_shellpair(1, s->df_shells+i, 1, s->df_shells+unit_shell_id, pair, s->screen_method);
        s->shellpair_memsize += (double) pair->memsize;
    }
    
    // Group density fitting shells by AM
    s->df_am_shell_id   = (int*) malloc(sizeof(int) * df_nshell);
    s->df_am_shell_spos = (int*) malloc(sizeof(int) * (s->df_max_am + 2));
    s->df_am_shell_num  = (int*) malloc(sizeof(int) * (s->df_max_am + 1));
    memset(s->df_am_shell_num, 0, sizeof(int) * (s->df_max_am + 1));
    for (int i = 0; i < df_nshell; i++)
    {
        int am = s->df_shells[i].am;
        s->df_am_shell_num[am]++;
    }
    memset(s->df_am_shell_spos, 0, sizeof(int) * (s->df_max_am + 2));
    for (int i = 1; i <= s->df_max_am + 1; i++)
        s->df_am_shell_spos[i] = s->df_am_shell_spos[i - 1] + s->df_am_shell_num[i - 1];
    memset(s->df_am_shell_num, 0, sizeof(int) * (s->df_max_am + 1));
    for (int i = 0; i < df_nshell; i++)
    {
        int am = s->df_shells[i].am;
        int group_pos = s->df_am_shell_spos[am] + s->df_am_shell_num[am];
        s->df_am_shell_id[group_pos] = i;
        s->df_am_shell_num[am]++;
    }
}

void CMS_Simint_free_DF_shellpairs(Simint_t simint)
{
    int df_nshell = simint->df_nshell;
    
    simint_free_shells(df_nshell + 1, simint->df_shells);
    simint_free_multi_shellpairs(df_nshell, simint->df_shellpairs);
    free(simint->df_shells);
    free(simint->df_shellpairs);
    free(simint->df_am_shell_id);
    free(simint->df_am_shell_spos);
    free(simint->df_am_shell_num);
    
    simint->df_shells        = NULL;
    simint->df_shellpairs    = NULL;
    simint->df_am_shell_id   = NULL;
    simint->df_am_shell_spos = NULL;
    simint->df_am_shell_num  = NULL;
}

void CMS_Simint_destroy(Simint_t simint, int show_stat)
{
    // Generate final statistic info
    double sum_msp = 0, sum_nprim = 0;
    double total_prim = 0, unscreened_prim = 0;
    double total_vec  = 0, unscreened_vec  = 0;
    for (int i = 0; i < simint->nthread; i++)
    {
        sum_msp    += (double) simint->num_multi_shellpairs[i];
        sum_nprim  += (double) simint->sum_nprim[i];
        total_prim      += simint->num_screened_prim[i] + simint->num_unscreened_prim[i];
        unscreened_prim += simint->num_unscreened_prim[i];
        total_vec       += simint->num_screened_vec[i] + simint->num_unscreened_vec[i];
        unscreened_vec  += simint->num_unscreened_vec[i];
    }
    double avg_nprim = sum_nprim / sum_msp;
    double prim_unscreen_ratio = unscreened_prim / total_prim;
    double vec_unscreen_ratio  = unscreened_vec  / total_vec;
    
    // Print timer and statistic info
    if (show_stat)
    {
        printf(
            "Timer: Simint setup, Simint ERI actual, Fock mat accum. = %lf, %lf, %lf sec\n", 
            simint->ostei_setup, simint->ostei_actual, simint->fock_update_F
        );
        printf(
            "Simint statistic: avg. ket-side nprim, prim unscreened ratio, SIMD unscreened ratio = %.1lf, %.1lf %%, %.1lf %%\n",
            avg_nprim, prim_unscreen_ratio * 100.0, vec_unscreen_ratio * 100.0
        );
    }

    // Free shell pair info
    int nshell = simint->nshell;
    int df_nshell = simint->df_nshell;
    if (simint->df_shells != NULL)
    {
        simint_free_shells(df_nshell + 1, simint->df_shells);
        simint_free_multi_shellpairs(df_nshell, simint->df_shellpairs);
    }
    simint_free_shells(nshell, simint->shells);
    simint_free_multi_shellpairs(nshell * nshell, simint->shellpairs);

    // Free memory
    free(simint->shellpairs);
    free(simint->shells);
    free(simint->df_shellpairs);
    free(simint->df_shells);
    free(simint->df_am_shell_id);
    free(simint->df_am_shell_spos);
    free(simint->df_am_shell_num);
    _mm_free(simint->workbuf);
    _mm_free(simint->outbuf);
    free(simint->num_multi_shellpairs);
    free(simint->sum_nprim);
    free(simint->num_screened_prim);
    free(simint->num_unscreened_prim);
    free(simint->num_screened_vec);
    free(simint->num_unscreened_vec);
    free(simint);

    simint_finalize();
}

int  CMS_Simint_get_sp_AM_idx(Simint_t simint, int P, int Q)
{
    shell_t shells = simint->shells;
    return shells[P].am * ((_SIMINT_OSTEI_MAXAM) + 1) + shells[Q].am;
}

double CMS_Simint_get_DF_sp_scrval(Simint_t simint, int i)
{
    multi_sp_t pair;
    pair = &simint->df_shellpairs[i];
    return pair->screen_max;
}

void CMS_Simint_create_multi_sp(void **multi_sp_)
{
    multi_sp_t multi_sp;
    multi_sp = (multi_sp_t) malloc(sizeof(multi_sp_s));
    CMS_ASSERT(multi_sp != NULL);
    // Need not to worry about memory allocation, it will be handled later
    simint_initialize_multi_shellpair(multi_sp);
    *multi_sp_ = multi_sp;
}

void CMS_Simint_free_multi_sp(void *multi_sp)
{
    CMS_ASSERT(multi_sp != NULL);
    simint_free_multi_shellpair(multi_sp);
    free(multi_sp);
}

static void CMS_Simint_fill_multi_sp_list(
    Simint_t simint, int npair, int *P_list, int *Q_list, 
    multi_sp_t multi_sp
)
{
    // Put the original multi_shellpairs corresponding to the shell
    // pairs (P_list[i], Q_list[i]) into the list
    multi_sp_t Pin[_SIMINT_NSHELL_SIMD];
    for (int ipair = 0; ipair < npair; ipair++)
    {
        int P = P_list[ipair];
        int Q = Q_list[ipair];
        Pin[ipair] = &simint->shellpairs[P * simint->nshell + Q];
    }
    
    // Reset output multi_sp and copy from existing multi_shellpairs.
    // simint_cat_multi_shellpair() will check and allocate memory for output
    multi_sp->nprim = 0;
    simint_cat_shellpairs(
        npair, (const struct simint_multi_shellpair **) Pin, 
        multi_sp, simint->screen_method
    );
}

void CMS_Simint_calc_pair_Hcore(
    BasisSet_t basis, Simint_t simint, int tid,
    int A, int B, double **integrals, int *nint
)
{
    int size, ret;
    struct simint_shell *shells = simint->shells;

    size = NCART(shells[A].am) * NCART(shells[B].am);

    double *workbuf = &simint->workbuf[tid * simint->workmem_per_thread];
    ret = simint_compute_ke(&shells[A], &shells[B], workbuf);
    CMS_ASSERT(ret == 1);

    double *output_buff = &simint->outbuf[tid * simint->outmem_per_thread];
    ret = simint_compute_potential(
        basis->natoms, basis->charge, basis->xn, basis->yn, basis->zn,
       &shells[A], &shells[B], output_buff
    );
    CMS_ASSERT(ret == 1);
    
    for (int i = 0; i < size; i++) output_buff[i] += workbuf[i];

    *integrals = output_buff;
    *nint = size;
}

void CMS_Simint_calc_pair_ovlp(
    Simint_t simint, int tid, int A, int B, 
    double **integrals, int *nint
)
{
    int size, ret;
    struct simint_shell *shells = simint->shells;
    double *output_buff = &simint->outbuf[tid*simint->outmem_per_thread];
    ret = simint_compute_overlap(&shells[A], &shells[B], output_buff);
    CMS_ASSERT(ret == 1);
    size = NCART(shells[A].am) * NCART(shells[B].am);
    *integrals = output_buff;
    *nint = size;
}

void CMS_Simint_calc_shellquartet(
    Simint_t simint, int tid, int M, int N, 
    int P, int Q, double **ERI, int *nint
)
{
    double setup_start, setup_end, ostei_start, ostei_end;

    if (tid == 0) setup_start = CMS_get_walltime_sec();

    int nshell = simint->nshell;
    multi_sp_t bra_pair = &simint->shellpairs[M * nshell + N];
    multi_sp_t ket_pair = &simint->shellpairs[P * nshell + Q];
    
    simint->num_multi_shellpairs[tid] += 1.0;
    simint->sum_nprim[tid] += (double) ket_pair->nprim;

    if (tid == 0) 
    {
        setup_end   = CMS_get_walltime_sec();
        ostei_start = CMS_get_walltime_sec();
    }
    
    double *work_buff   = &simint->workbuf[tid * simint->workmem_per_thread];
    double *output_buff = &simint->outbuf [tid * simint->outmem_per_thread];
    int ret = simint_compute_eri(
        bra_pair, ket_pair, simint->screen_tol,
        work_buff, output_buff
    );
    
    if (tid == 0) ostei_end = CMS_get_walltime_sec();
    
    int ERI_size;
    if (ret < 0) 
    {
        ERI_size = 0; // Return zero ERI_size to caller; output buffer is not initialized
    } else {
        CMS_ASSERT(ret == 1); // Single shell quartet
        shell_t shells = simint->shells;
        ERI_size = NCART(shells[M].am) * NCART(shells[N].am)
                 * NCART(shells[P].am) * NCART(shells[Q].am);
    }

    *ERI  = output_buff;
    *nint = ERI_size;
    
    double *prim_screen_stat_info = *ERI + ERI_size;
    simint->num_unscreened_prim[tid] += prim_screen_stat_info[0];
    simint->num_screened_prim[tid]   += prim_screen_stat_info[1];
    simint->num_unscreened_vec[tid]  += prim_screen_stat_info[2];
    simint->num_screened_vec[tid]    += prim_screen_stat_info[3];

    if (tid == 0)
    {
        simint->ostei_setup  += setup_end - setup_start;
        simint->ostei_actual += ostei_end - ostei_start;
    }
}

void CMS_Simint_calc_shellquartet_batch(
    Simint_t simint, int tid, int M, int N, int npair, int *P_list, 
    int *Q_list, double **batch_ERI, int *batch_nint, void **multi_sp_
)
{
    double setup_start, setup_end, ostei_start, ostei_end;
    
    if (tid == 0) setup_start = CMS_get_walltime_sec();

    multi_sp_t bra_pair  = &simint->shellpairs[M * simint->nshell + N];
    multi_sp_t ket_pairs = (multi_sp_t) *multi_sp_;
    
    CMS_Simint_fill_multi_sp_list(simint, npair, P_list, Q_list, ket_pairs);
    
    simint->num_multi_shellpairs[tid] += 1.0;
    simint->sum_nprim[tid] += (double) ket_pairs->nprim;
    
    if (tid == 0) 
    {
        setup_end   = CMS_get_walltime_sec();
        ostei_start = CMS_get_walltime_sec();
    }
    
    double *work_buff   = &simint->workbuf[tid * simint->workmem_per_thread];
    double *output_buff = &simint->outbuf [tid * simint->outmem_per_thread];
    int ret = simint_compute_eri(
        bra_pair, ket_pairs, simint->screen_tol,
        work_buff, output_buff
    );
    
    if (tid == 0) ostei_end = CMS_get_walltime_sec();
    
    int ERI_size;
    if (ret <= 0)
    {
        ERI_size = 0; // Return zero ERI_size to caller; output buffer is not initialized
    } else {
        CMS_ASSERT(ret == npair);
        shell_t shells = simint->shells;
        int P = P_list[0], Q = Q_list[0];
        ERI_size = NCART(shells[M].am) * NCART(shells[N].am) 
                 * NCART(shells[P].am) * NCART(shells[Q].am);
    }
    
    // Shells in P_list[] have same AM, shells in Q_list[] have same AM,
    // The result sizes for each quartets are the same
    *batch_ERI  = output_buff;
    *batch_nint = ERI_size;
    
    double *prim_screen_stat_info = *batch_ERI + ERI_size * npair;
    simint->num_unscreened_prim[tid] += prim_screen_stat_info[0];
    simint->num_screened_prim[tid]   += prim_screen_stat_info[1];
    simint->num_unscreened_vec[tid]  += prim_screen_stat_info[2];
    simint->num_screened_vec[tid]    += prim_screen_stat_info[3];
    
    if (tid == 0)
    {
        simint->ostei_setup  += setup_end - setup_start;
        simint->ostei_actual += ostei_end - ostei_start;
    }
}

static void CMS_Simint_fill_DF_multi_sp_list(
    Simint_t simint, int npair, int *P_list, 
    struct simint_multi_shellpair *multi_sp
)
{
    // Put the original multi_shellpairs corresponding to the shell
    // pairs (P_list[i], Q_list[i]) into the list
    multi_sp_t Pin[_SIMINT_NSHELL_SIMD];
    for (int ipair = 0; ipair < npair; ipair++)
    {
        int P = P_list[ipair];
        Pin[ipair] = &simint->df_shellpairs[P];
    }
    
    // Reset output multi_sp and copy from existing multi_shellpairs.
    // simint_cat_multi_shellpair() will check and allocate memory for output
    multi_sp->nprim = 0;
    simint_cat_shellpairs(
        npair, (const struct simint_multi_shellpair **) Pin, 
        multi_sp, simint->screen_method
    );
}

void CMS_Simint_calc_DF_shellpair(
    Simint_t simint, int tid, int M, int N,
    double **integrals, int *nint
)
{
    double setup_start, setup_end, ostei_start, ostei_end;

    if (tid == 0) setup_start = CMS_get_walltime_sec();

    multi_sp_t bra_pair = &simint->df_shellpairs[M];
    multi_sp_t ket_pair = &simint->df_shellpairs[N];
    
    simint->num_multi_shellpairs[tid] += 1.0;
    simint->sum_nprim[tid] += (double) ket_pair->nprim;

    if (tid == 0) 
    {
        setup_end   = CMS_get_walltime_sec();
        ostei_start = CMS_get_walltime_sec();
    }
    
    double *work_buff   = &simint->workbuf[tid * simint->workmem_per_thread];
    double *output_buff = &simint->outbuf [tid * simint->outmem_per_thread];
    int ret = simint_compute_eri(
        bra_pair, ket_pair, simint->screen_tol,
        work_buff, output_buff
    );
    
    if (tid == 0) ostei_end = CMS_get_walltime_sec();
    
    int ERI_size;
    if (ret < 0) 
    {
        ERI_size = 0; // Return zero ERI_size to caller; shell quartet is screened 
    } else {
        CMS_ASSERT(ret == 1);
        shell_t df_shells = simint->df_shells;
        ERI_size = NCART(df_shells[M].am) * NCART(df_shells[N].am);
    }

    *integrals = output_buff;
    *nint = ERI_size;
    
    double *prim_screen_stat_info = *integrals + ERI_size;
    simint->num_unscreened_prim[tid] += prim_screen_stat_info[0];
    simint->num_screened_prim[tid]   += prim_screen_stat_info[1];
    simint->num_unscreened_vec[tid]  += prim_screen_stat_info[2];
    simint->num_screened_vec[tid]    += prim_screen_stat_info[3];

    if (tid == 0)
    {
        simint->ostei_setup  += setup_end - setup_start;
        simint->ostei_actual += ostei_end - ostei_start;
    }
}

void CMS_Simint_calc_DF_shellquartet_batch(
    Simint_t simint, int tid, int M, int N, int npair, int *P_list, 
    double **batch_ERI, int *batch_nint, void **multi_sp_
)
{
    double setup_start, setup_end, ostei_start, ostei_end;

    if (tid == 0) setup_start = CMS_get_walltime_sec();

    multi_sp_t bra_pair  = &simint->shellpairs[M * simint->nshell + N];
    multi_sp_t ket_pairs = (multi_sp_t) *multi_sp_;
    
    CMS_Simint_fill_DF_multi_sp_list(simint, npair, P_list, ket_pairs);
    
    simint->num_multi_shellpairs[tid] += 1.0;
    simint->sum_nprim[tid] += (double) ket_pairs->nprim;
    
    if (tid == 0) 
    {
        setup_end   = CMS_get_walltime_sec();
        ostei_start = CMS_get_walltime_sec();
    }
    
    double *work_buff   = &simint->workbuf[tid * simint->workmem_per_thread];
    double *output_buff = &simint->outbuf [tid * simint->outmem_per_thread];
    int ret = simint_compute_eri(
        bra_pair, ket_pairs, simint->screen_tol,
        work_buff, output_buff
    );
    
    if (tid == 0) ostei_end = CMS_get_walltime_sec();
    
    int ERI_size;
    if (ret <= 0)
    {
        ERI_size = 0; // Return zero ERI_size to caller; output buffer is not initialized
    } else {
        CMS_ASSERT(ret == npair);
        shell_t shells = simint->shells;
        shell_t df_shells = simint->df_shells;
        int P = P_list[0];
        ERI_size = NCART(shells[M].am) * NCART(shells[N].am) * NCART(df_shells[P].am);
    }
    
    // Shells in P_list[] have same AM, shells in Q_list[] have same AM,
    // The result sizes for each quartets are the same
    *batch_ERI  = output_buff;
    *batch_nint = ERI_size;
    
    double *prim_screen_stat_info = *batch_ERI + ERI_size * npair;
    simint->num_unscreened_prim[tid] += prim_screen_stat_info[0];
    simint->num_screened_prim[tid]   += prim_screen_stat_info[1];
    simint->num_unscreened_vec[tid]  += prim_screen_stat_info[2];
    simint->num_screened_vec[tid]    += prim_screen_stat_info[3];
    
    if (tid == 0)
    {
        simint->ostei_setup  += setup_end - setup_start;
        simint->ostei_actual += ostei_end - ostei_start;
    }
}

void CMS_Simint_add_accF_timer(Simint_t simint, double sec)
{
    simint->fock_update_F += sec;
}

void CMS_Simint_reset_stat_info(Simint_t simint)
{
    int stat_info_size = sizeof(double) * simint->nthread;
    memset(simint->num_multi_shellpairs, 0, stat_info_size);
    memset(simint->sum_nprim,            0, stat_info_size);
    memset(simint->num_screened_prim,    0, stat_info_size);
    memset(simint->num_unscreened_prim,  0, stat_info_size);
    memset(simint->num_screened_vec,     0, stat_info_size);
    memset(simint->num_unscreened_vec,   0, stat_info_size);
}
