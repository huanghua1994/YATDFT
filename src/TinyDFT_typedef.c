#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <libgen.h>
#include <float.h>
#include <time.h>
#include <omp.h>

#ifdef USE_LIBXC
#include <xc.h>
#endif

#include "libCMS.h"
#include "utils.h"
#include "TinyDFT_typedef.h"
#include "build_HF_mat.h"
#include "build_Dmat.h"
#include "CDIIS.h"

// Compute screening value of each shell pair and find all 
// unique shell pairs that survive Schwarz screening
// Input parameter:
//   TinyDFT : Initialized TinyDFT structure
// Output parameters:
//   TinyDFT : TinyDFT structure with screening info
static void TinyDFT_screen_shell_quartets(TinyDFT_t TinyDFT);

void TinyDFT_init(TinyDFT_t *TinyDFT_, char *bas_fname, char *xyz_fname)
{
    TinyDFT_t TinyDFT = (TinyDFT_t) malloc(sizeof(struct TinyDFT_struct));
    assert(TinyDFT != NULL);
    
    double st = get_wtime_sec();
    
    TinyDFT->nthread = omp_get_max_threads();
    
    // Reset statistic info
    TinyDFT->mem_size       = 0.0;
    TinyDFT->init_time      = 0.0;
    TinyDFT->S_Hcore_time   = 0.0;
    TinyDFT->shell_scr_time = 0.0;
    
    // Load basis set and molecule from input 
    CMS_createBasisSet(&(TinyDFT->basis));
    CMS_loadChemicalSystem(TinyDFT->basis, bas_fname, xyz_fname);
    int maxAM = CMS_getMaxMomentum(TinyDFT->basis);
    TinyDFT->bas_name      = basename(bas_fname);
    TinyDFT->mol_name      = basename(xyz_fname);
    TinyDFT->natom         = CMS_getNumAtoms   (TinyDFT->basis);
    TinyDFT->nshell        = CMS_getNumShells  (TinyDFT->basis);
    TinyDFT->nbf           = CMS_getNumFuncs   (TinyDFT->basis);
    TinyDFT->n_occ         = CMS_getNumOccOrb  (TinyDFT->basis);
    TinyDFT->charge        = CMS_getTotalCharge(TinyDFT->basis);
    TinyDFT->electron      = CMS_getNneutral   (TinyDFT->basis);
    TinyDFT->num_total_sp  = TinyDFT->nshell * TinyDFT->nshell;
    TinyDFT->num_valid_sp  = (TinyDFT->nshell + 1) * TinyDFT->nshell / 2;
    TinyDFT->mat_size      = TinyDFT->nbf * TinyDFT->nbf;
    TinyDFT->max_dim       = (maxAM + 1) * (maxAM + 2) / 2;
    TinyDFT->prim_scrtol   = 1e-14;
    TinyDFT->shell_scrtol2 = 1e-11 * 1e-11;
    TinyDFT->E_nuc_rep     = CMS_getNucEnergy(TinyDFT->basis);
    printf("Job information:\n");
    printf("    basis set       = %s\n", TinyDFT->bas_name);
    printf("    molecule        = %s\n", TinyDFT->mol_name);
    printf("    atoms           = %d\n", TinyDFT->natom);
    printf("    shells          = %d\n", TinyDFT->nshell);
    printf("    basis functions = %d\n", TinyDFT->nbf);
    printf("    occupied orbits = %d\n", TinyDFT->n_occ);
    printf("    charge          = %d\n", TinyDFT->charge);
    printf("    electrons       = %d\n", TinyDFT->electron);
    int nthread      = TinyDFT->nthread;
    int nshell       = TinyDFT->nshell;
    int nbf          = TinyDFT->nbf;
    int n_occ        = TinyDFT->n_occ;
    int num_total_sp = TinyDFT->num_total_sp;
    int num_valid_sp = TinyDFT->num_valid_sp;
    
    // Allocate memory for ERI info arrays for direct approach
    CMS_Simint_init(TinyDFT->basis, &(TinyDFT->simint), nthread, TinyDFT->prim_scrtol);
    TinyDFT->valid_sp_lid   = (int*)    malloc_aligned(INT_MSIZE * num_valid_sp, 64);
    TinyDFT->valid_sp_rid   = (int*)    malloc_aligned(INT_MSIZE * num_valid_sp, 64);
    TinyDFT->shell_bf_sind  = (int*)    malloc_aligned(INT_MSIZE * (nshell + 1), 64);
    TinyDFT->shell_bf_num   = (int*)    malloc_aligned(INT_MSIZE * nshell,       64);
    TinyDFT->sp_scrval      = (double*) malloc_aligned(DBL_MSIZE * num_total_sp, 64);
    TinyDFT->bf_pair_scrval = (double*) malloc_aligned(DBL_MSIZE * nbf * nbf,    64);
    assert(TinyDFT->valid_sp_lid  != NULL);
    assert(TinyDFT->valid_sp_rid  != NULL);
    assert(TinyDFT->shell_bf_sind != NULL);
    assert(TinyDFT->shell_bf_num  != NULL);
    assert(TinyDFT->sp_scrval     != NULL);
    TinyDFT->mem_size += (double) (INT_MSIZE * 2 * TinyDFT->num_valid_sp);
    TinyDFT->mem_size += (double) (INT_MSIZE * (2 * nshell + 1));
    TinyDFT->mem_size += (double) (DBL_MSIZE * num_total_sp);
    TinyDFT->mem_size += (double) (DBL_MSIZE * nbf * nbf);
    for (int i = 0; i < nshell; i++)
    {
        TinyDFT->shell_bf_sind[i] = CMS_getFuncStartInd(TinyDFT->basis, i);
        TinyDFT->shell_bf_num[i]  = CMS_getShellDim    (TinyDFT->basis, i);
    }
    TinyDFT->shell_bf_sind[nshell] = nbf;
    
    // Molecular system and ERI info for density fitting will
    // be allocated later if needed
    TinyDFT->df_shell_bf_sind = NULL;
    TinyDFT->df_shell_bf_num  = NULL;
    TinyDFT->bf_pair_mask     = NULL;
    TinyDFT->bf_pair_j        = NULL;
    TinyDFT->bf_pair_diag     = NULL;
    TinyDFT->bf_mask_displs   = NULL;
    TinyDFT->df_sp_scrval     = NULL;
    TinyDFT->df_basis         = NULL;
    
    // Flattened Gaussian basis function and atom info used only 
    // in XC calculation will be allocated if needed
    TinyDFT->atom_idx  = NULL;
    TinyDFT->bf_nprim  = NULL;
    TinyDFT->atom_xyz  = NULL;
    TinyDFT->bf_coef   = NULL;
    TinyDFT->bf_alpha  = NULL;
    TinyDFT->bf_exp    = NULL;
    TinyDFT->bf_center = NULL;
    
    // Allocate memory for matrices and arrays used only in build_HF_mat
    size_t mat_msize          = DBL_MSIZE * TinyDFT->mat_size;
    size_t MN_strip_msize     = DBL_MSIZE * TinyDFT->max_dim * nbf;
    size_t max_buf_entry_size = TinyDFT->max_dim * TinyDFT->max_dim;
    size_t total_buf_size     = max_buf_entry_size * 6 * nthread;
    TinyDFT->max_JKacc_buf = max_buf_entry_size * 6;
    TinyDFT->blk_mat_ptr   = (int*)    malloc_aligned(INT_MSIZE * TinyDFT->num_total_sp, 64);
    TinyDFT->Mpair_flag    = (int*)    malloc_aligned(INT_MSIZE * nshell * nthread,      64);
    TinyDFT->Npair_flag    = (int*)    malloc_aligned(INT_MSIZE * nshell * nthread,      64);
    TinyDFT->J_blk_mat     = (double*) malloc_aligned(mat_msize,                         64);
    TinyDFT->K_blk_mat     = (double*) malloc_aligned(mat_msize,                         64);
    TinyDFT->D_blk_mat     = (double*) malloc_aligned(mat_msize,                         64);
    TinyDFT->JKacc_buf     = (double*) malloc_aligned(DBL_MSIZE * total_buf_size,        64);
    TinyDFT->FM_strip_buf  = (double*) malloc_aligned(MN_strip_msize * nthread,          64);
    TinyDFT->FN_strip_buf  = (double*) malloc_aligned(MN_strip_msize * nthread,          64);
    assert(TinyDFT->blk_mat_ptr  != NULL);
    assert(TinyDFT->Mpair_flag   != NULL);
    assert(TinyDFT->Npair_flag   != NULL);
    assert(TinyDFT->J_blk_mat    != NULL);
    assert(TinyDFT->K_blk_mat    != NULL);
    assert(TinyDFT->D_blk_mat    != NULL);
    assert(TinyDFT->JKacc_buf    != NULL);
    assert(TinyDFT->FM_strip_buf != NULL);
    assert(TinyDFT->FN_strip_buf != NULL);
    TinyDFT->mem_size += (double) (INT_MSIZE * TinyDFT->num_total_sp);
    TinyDFT->mem_size += (double) (2 * INT_MSIZE * nshell * nthread);
    TinyDFT->mem_size += (double) (3 * mat_msize);
    TinyDFT->mem_size += (double) (2 * MN_strip_msize * nthread);
    TinyDFT->mem_size += (double) (DBL_MSIZE * total_buf_size);
    int pos = 0, idx = 0;
    for (int i = 0; i < nshell; i++)
    {
        for (int j = 0; j < nshell; j++)
        {
            TinyDFT->blk_mat_ptr[idx] = pos;
            pos += TinyDFT->shell_bf_num[i] * TinyDFT->shell_bf_num[j];
            idx++;
        }
    }
    
    // Matrices and arrays used in XC functional calculation will 
    // be allocated later if needed
    TinyDFT->int_grid   = NULL;
    TinyDFT->phi        = NULL;
    TinyDFT->rho        = NULL;
    TinyDFT->exc        = NULL;
    TinyDFT->vxc        = NULL;
    TinyDFT->vsigma     = NULL;
    TinyDFT->XC_workbuf = NULL;
    TinyDFT->xf_impl    = 1;
    TinyDFT->cf_impl    = 1;
    
    // Allocate memory for matrices used in multiple modules
    TinyDFT->tmp_mat = (double*) malloc_aligned(mat_msize * 4, 64);
    assert(TinyDFT->tmp_mat != NULL);
    TinyDFT->mem_size += (double) (mat_msize * 4);
    
    // Allocate memory for matrices and arrays used only in build_Dmat
    TinyDFT->ev_idx = (int*)    malloc_aligned(INT_MSIZE * nbf, 64);
    TinyDFT->eigval = (double*) malloc_aligned(DBL_MSIZE * nbf, 64);
    assert(TinyDFT->ev_idx != NULL);
    assert(TinyDFT->eigval != NULL);
    TinyDFT->mem_size += (double) ((DBL_MSIZE + INT_MSIZE) * nbf);
    
    // Allocate memory for matrices and arrays used only in CDIIS
    int MAX_DIIS_1 = MAX_DIIS + 1;
    size_t DIIS_row_msize = DBL_MSIZE * MAX_DIIS_1;
    TinyDFT->F0_mat    = (double*) malloc_aligned(mat_msize * MAX_DIIS,        64);
    TinyDFT->R_mat     = (double*) malloc_aligned(mat_msize * MAX_DIIS,        64);
    TinyDFT->B_mat     = (double*) malloc_aligned(DIIS_row_msize * MAX_DIIS_1, 64);
    TinyDFT->FDS_mat   = (double*) malloc_aligned(mat_msize,                   64);
    TinyDFT->DIIS_rhs  = (double*) malloc_aligned(DIIS_row_msize,              64);
    TinyDFT->DIIS_ipiv = (int*)    malloc_aligned(INT_MSIZE * MAX_DIIS_1,      64);
    assert(TinyDFT->F0_mat    != NULL);
    assert(TinyDFT->R_mat     != NULL);
    assert(TinyDFT->B_mat     != NULL);
    assert(TinyDFT->DIIS_rhs  != NULL);
    assert(TinyDFT->DIIS_ipiv != NULL);
    TinyDFT->mem_size += MAX_DIIS * 2 * (double) mat_msize;
    TinyDFT->mem_size += (double) DIIS_row_msize * (MAX_DIIS + 2);
    TinyDFT->mem_size += (double) (INT_MSIZE * MAX_DIIS_1);
    TinyDFT->mem_size += (double) mat_msize;
    // Must initialize F0 and R as 0 
    memset(TinyDFT->F0_mat, 0, mat_msize * MAX_DIIS);
    memset(TinyDFT->R_mat,  0, mat_msize * MAX_DIIS);
    TinyDFT->DIIS_len = 0;
    // Initialize B_mat
    for (int i = 0; i < MAX_DIIS_1 * MAX_DIIS_1; i++) TinyDFT->B_mat[i] = -1.0;
    for (int i = 0; i < MAX_DIIS_1; i++) TinyDFT->B_mat[i * MAX_DIIS_1 + i] = 0.0;
    TinyDFT->DIIS_bmax_id = 0;
    TinyDFT->DIIS_bmax    = -DBL_MAX;

    // Allocate memory for matrices and arrays used only in SCF iterations
    TinyDFT->E_tol     = 1e-10;
    TinyDFT->Hcore_mat = (double*) malloc_aligned(mat_msize, 64);
    TinyDFT->S_mat     = (double*) malloc_aligned(mat_msize, 64);
    TinyDFT->X_mat     = (double*) malloc_aligned(mat_msize, 64);
    TinyDFT->J_mat     = (double*) malloc_aligned(mat_msize, 64);
    TinyDFT->K_mat     = (double*) malloc_aligned(mat_msize, 64);
    TinyDFT->XC_mat    = (double*) malloc_aligned(mat_msize, 64);
    TinyDFT->F_mat     = (double*) malloc_aligned(mat_msize, 64);
    TinyDFT->D_mat     = (double*) malloc_aligned(mat_msize, 64);
    TinyDFT->C_mat     = (double*) malloc_aligned(mat_msize, 64);
    TinyDFT->Cocc_mat  = (double*) malloc_aligned(mat_msize, 64);
    assert(TinyDFT->Hcore_mat != NULL);
    assert(TinyDFT->S_mat     != NULL);
    assert(TinyDFT->X_mat     != NULL);
    assert(TinyDFT->J_mat     != NULL);
    assert(TinyDFT->K_mat     != NULL);
    assert(TinyDFT->XC_mat    != NULL);
    assert(TinyDFT->F_mat     != NULL);
    assert(TinyDFT->D_mat     != NULL);
    assert(TinyDFT->C_mat     != NULL);
    assert(TinyDFT->Cocc_mat  != NULL);
    TinyDFT->mem_size += (double) (10 * mat_msize);
    memset(TinyDFT->Cocc_mat, 0, DBL_MSIZE * n_occ * nbf);

    // Tensors and matrices used only in build_JKDF will 
    // be allocated later if needed
    TinyDFT->mat_K_m      = NULL;
    TinyDFT->mat_K_n      = NULL;
    TinyDFT->mat_K_k      = NULL;
    TinyDFT->mat_K_lda    = NULL;
    TinyDFT->mat_K_ldb    = NULL;
    TinyDFT->mat_K_ldc    = NULL;
    TinyDFT->mat_K_beta   = NULL;
    TinyDFT->mat_K_alpha  = NULL;
    TinyDFT->pqA          = NULL;
    TinyDFT->Jpq          = NULL;
    TinyDFT->df_tensor    = NULL;
    TinyDFT->temp_J       = NULL;
    TinyDFT->temp_K       = NULL;
    TinyDFT->mat_K_a      = NULL;
    TinyDFT->mat_K_b      = NULL;
    TinyDFT->mat_K_c      = NULL;
    TinyDFT->mat_K_transa = NULL;
    TinyDFT->mat_K_transb = NULL;

    double et = get_wtime_sec();
    TinyDFT->init_time = et - st;
    
    // Print memory usage and time consumption
    printf("TinyDFT memory allocation and initialization over, elapsed time = %.3lf (s)\n", TinyDFT->init_time);
    
    TinyDFT_screen_shell_quartets(TinyDFT);
    
    *TinyDFT_ = TinyDFT;
}

void TinyDFT_destroy(TinyDFT_t *_TinyDFT)
{
    TinyDFT_t TinyDFT = *_TinyDFT;
    assert(TinyDFT != NULL);
    
    printf("TinyDFT total memory usage = %.2lf MB\n", TinyDFT->mem_size / 1048576.0);
    
    // Free ERI info arrays for direct approach
    free_aligned(TinyDFT->valid_sp_lid);
    free_aligned(TinyDFT->valid_sp_rid);
    free_aligned(TinyDFT->shell_bf_sind);
    free_aligned(TinyDFT->shell_bf_num);
    free_aligned(TinyDFT->sp_scrval);
    free_aligned(TinyDFT->bf_pair_scrval);
    
    // Free ERI info arrays for density fitting
    free_aligned(TinyDFT->df_shell_bf_sind);
    free_aligned(TinyDFT->df_shell_bf_num);
    free_aligned(TinyDFT->bf_pair_mask);
    free_aligned(TinyDFT->bf_pair_j);
    free_aligned(TinyDFT->bf_pair_diag);
    free_aligned(TinyDFT->bf_mask_displs);
    free_aligned(TinyDFT->df_sp_scrval);
    
    // Free flattened Gaussian basis function and atom info used only 
    // in XC calculation
    free_aligned(TinyDFT->atom_idx);
    free_aligned(TinyDFT->bf_nprim);
    free_aligned(TinyDFT->atom_xyz);
    free_aligned(TinyDFT->bf_coef);
    free_aligned(TinyDFT->bf_alpha);
    free_aligned(TinyDFT->bf_exp);
    free_aligned(TinyDFT->bf_center);
    
    // Free matrices and temporary arrays used only in build_HF_mat
    free_aligned(TinyDFT->blk_mat_ptr);
    free_aligned(TinyDFT->Mpair_flag);
    free_aligned(TinyDFT->Npair_flag);
    free_aligned(TinyDFT->J_blk_mat);
    free_aligned(TinyDFT->K_blk_mat);
    free_aligned(TinyDFT->D_blk_mat);
    free_aligned(TinyDFT->JKacc_buf);
    free_aligned(TinyDFT->FM_strip_buf);
    free_aligned(TinyDFT->FN_strip_buf);
    
    // Free matrices and arrays used in XC functional calculation
    free(TinyDFT->int_grid);
    free_aligned(TinyDFT->phi);
    free_aligned(TinyDFT->rho);
    free_aligned(TinyDFT->exc);
    free_aligned(TinyDFT->vxc);
    free_aligned(TinyDFT->vsigma);
    free_aligned(TinyDFT->XC_workbuf);
    #ifdef USE_LIBXC
    if (TinyDFT->xf_impl == 0) xc_func_end(&TinyDFT->libxc_xf);
    if (TinyDFT->cf_impl == 0) xc_func_end(&TinyDFT->libxc_cf);
    #endif
    
    // Free matrices used in multiple modules
    free_aligned(TinyDFT->tmp_mat);
    
    // Free matrices and arrays used only in build_Dmat
    free_aligned(TinyDFT->ev_idx);
    free_aligned(TinyDFT->eigval);
    
    // Free matrices and temporary arrays used only in CDIIS
    free_aligned(TinyDFT->F0_mat);
    free_aligned(TinyDFT->R_mat);
    free_aligned(TinyDFT->B_mat);
    free_aligned(TinyDFT->FDS_mat);
    free_aligned(TinyDFT->DIIS_rhs);
    free_aligned(TinyDFT->DIIS_ipiv);
    
    // Free matrices and temporary arrays used only in SCF
    free_aligned(TinyDFT->Hcore_mat);
    free_aligned(TinyDFT->S_mat);
    free_aligned(TinyDFT->F_mat);
    free_aligned(TinyDFT->D_mat);
    free_aligned(TinyDFT->J_mat);
    free_aligned(TinyDFT->K_mat);
    free_aligned(TinyDFT->X_mat);
    free_aligned(TinyDFT->C_mat);
    free_aligned(TinyDFT->Cocc_mat);
    
    // Free Tensors and matrices used only in build_JKDF
    free(TinyDFT->mat_K_m);
    free(TinyDFT->mat_K_n);
    free(TinyDFT->mat_K_k);
    free(TinyDFT->mat_K_lda);
    free(TinyDFT->mat_K_ldb);
    free(TinyDFT->mat_K_ldc);
    free(TinyDFT->mat_K_beta);
    free(TinyDFT->mat_K_alpha);
    free_aligned(TinyDFT->pqA);
    free_aligned(TinyDFT->Jpq);
    free_aligned(TinyDFT->df_tensor);
    free_aligned(TinyDFT->temp_J);
    free_aligned(TinyDFT->temp_K);
    free(TinyDFT->mat_K_a);
    free(TinyDFT->mat_K_b);
    free(TinyDFT->mat_K_c);
    free(TinyDFT->mat_K_transa);
    free(TinyDFT->mat_K_transb);

    // Free BasisSet_t and Simint_t object, print Simint_t object stat info
    CMS_destroyBasisSet(TinyDFT->basis);
    CMS_Simint_destroy(TinyDFT->simint, 1);
    
    free(TinyDFT);
    *_TinyDFT = NULL;
}

static int cmp_pair(int M1, int N1, int M2, int N2)
{
    if (M1 == M2) return (N1 < N2);
    else return (M1 < M2);
}

static void quickSort(int *M, int *N, int l, int r)
{
    int i = l, j = r, tmp;
    int mid_M = M[(i + j) / 2];
    int mid_N = N[(i + j) / 2];
    while (i <= j)
    {
        while (cmp_pair(M[i], N[i], mid_M, mid_N)) i++;
        while (cmp_pair(mid_M, mid_N, M[j], N[j])) j--;
        if (i <= j)
        {
            tmp = M[i]; M[i] = M[j]; M[j] = tmp;
            tmp = N[i]; N[i] = N[j]; N[j] = tmp;
            
            i++;  j--;
        }
    }
    if (i < r) quickSort(M, N, i, r);
    if (j > l) quickSort(M, N, l, j);
}

static void TinyDFT_screen_shell_quartets(TinyDFT_t TinyDFT)
{
    assert(TinyDFT != NULL);
    
    int    nshell          = TinyDFT->nshell;
    int    nbf             = TinyDFT->nbf;
    int    *shell_bf_num   = TinyDFT->shell_bf_num;
    int    *shell_bf_sind  = TinyDFT->shell_bf_sind;
    int    *valid_sp_lid   = TinyDFT->valid_sp_lid;
    int    *valid_sp_rid   = TinyDFT->valid_sp_rid;
    double shell_scrtol2   = TinyDFT->shell_scrtol2;
    double *sp_scrval      = TinyDFT->sp_scrval;
    double *bf_pair_scrval = TinyDFT->bf_pair_scrval;
    Simint_t simint        = TinyDFT->simint;
    
    double st = get_wtime_sec();
    
    // Compute screening values using Schwarz inequality
    double global_max_scrval = 0.0;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic) reduction(max:global_max_scrval)
        for (int M = 0; M < nshell; M++)
        {
            int dimM = shell_bf_num[M];
            int M_bf_idx = shell_bf_sind[M];
            for (int N = 0; N < nshell; N++)
            {
                int dimN = shell_bf_num[N];
                int N_bf_idx = shell_bf_sind[N];
                
                int nints;
                double *integrals;
                CMS_Simint_calc_shellquartet(simint, tid, M, N, M, N, &integrals, &nints);
                
                double maxval = 0.0;
                if (nints > 0)
                {
                    // Loop over all ERIs in a shell quartet and find the max value
                    for (int iM = 0; iM < dimM; iM++)
                    {
                        for (int iN = 0; iN < dimN; iN++)
                        {
                            int index = iN * (dimM * dimN * dimM + dimM) + iM * (dimN * dimM + 1); // Simint layout
                            double val = fabs(integrals[index]);
                            int bf_idx = (M_bf_idx + iM) * nbf + (N_bf_idx + iN);
                            bf_pair_scrval[bf_idx] = val;
                            if (val > maxval) maxval = val;
                        }
                    }
                }
                sp_scrval[M * nshell + N] = maxval;
                if (maxval > global_max_scrval) global_max_scrval = maxval;
            }
        }
    }
    
    // Reset Simint statistic info
    CMS_Simint_reset_stat_info(simint);
    
    // Generate unique shell pairs that survive Schwarz screening
    // eta is the threshold for screening a shell pair
    double eta = shell_scrtol2 / global_max_scrval;
    int num_valid_sp = 0;
    for (int M = 0; M < nshell; M++)
    {
        for (int N = 0; N < nshell; N++)
        {
            double MN_scrval = sp_scrval[M * nshell + N];
            // if sp_scrval * max_scrval < shell_scrtol2, for any given shell pair
            // (P,Q), (MN|PQ) is always < shell_scrtol2 and will be screened
            if (MN_scrval > eta)  
            {
                // Make {N_i} in (M, N_i) as continuous as possible to get better
                // memory access pattern and better performance
                if (N > M) continue;
                
                // We want AM(M) >= AM(N) to avoid HRR
                int MN_id = CMS_Simint_get_sp_AM_idx(simint, M, N);
                int NM_id = CMS_Simint_get_sp_AM_idx(simint, N, M);
                if (MN_id > NM_id)
                {
                    valid_sp_lid[num_valid_sp] = M;
                    valid_sp_rid[num_valid_sp] = N;
                } else {
                    valid_sp_lid[num_valid_sp] = N;
                    valid_sp_rid[num_valid_sp] = M;
                }
                num_valid_sp++;
            }
        }
    }
    TinyDFT->num_valid_sp = num_valid_sp;
    quickSort(valid_sp_lid, valid_sp_rid, 0, num_valid_sp - 1);
    
    double et = get_wtime_sec();
    TinyDFT->shell_scr_time = et - st;
    
    // Print runtime
    int num_total_sp = TinyDFT->num_total_sp;
    printf(
        "TinyDFT shell pair screening over, tol = %.2e, elapsed time = %.3lf (s)\n", 
        sqrt(shell_scrtol2), TinyDFT->shell_scr_time
    );
    printf(
        "Screened unique shell pairs: %d out of %d (sparsity = %.2lf%%)\n", 
        num_valid_sp, num_total_sp, 100.0 * (double) num_valid_sp / (double) num_total_sp
    );
}

