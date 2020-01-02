#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <libgen.h>
#include <omp.h>

#include "utils.h"
#include "libCMS.h"
#include "TinyDFT_typedef.h"
#include "setup_DF.h"

static void TinyDFT_load_DF_basis(TinyDFT_t TinyDFT, char *df_bas_fname, char *xyz_fname)
{
    int nthread = TinyDFT->nthread;
    CMS_createBasisSet(&(TinyDFT->df_basis));
    CMS_loadChemicalSystem(TinyDFT->df_basis, df_bas_fname, xyz_fname);
    CMS_Simint_setup_DF(TinyDFT->simint, TinyDFT->df_basis);
    TinyDFT->df_bas_name = basename(df_bas_fname);
    TinyDFT->df_nbf      = CMS_getNumFuncs (TinyDFT->df_basis);
    TinyDFT->df_nshell   = CMS_getNumShells(TinyDFT->df_basis);
    printf("Density fitting information:\n");
    printf("    DF basis set         = %s\n", TinyDFT->df_bas_name);
    printf("    # DF shells          = %d\n", TinyDFT->df_nshell);
    printf("    # DF basis functions = %d\n", TinyDFT->df_nbf);
    
    int nbf       = TinyDFT->nbf;
    int df_nshell = TinyDFT->df_nshell;
    int mat_size  = TinyDFT->mat_size;
    TinyDFT->df_shell_bf_sind = (int*)    ALIGN64B_MALLOC(INT_SIZE * (df_nshell + 1));
    TinyDFT->df_shell_bf_num  = (int*)    ALIGN64B_MALLOC(INT_SIZE * df_nshell);
    TinyDFT->bf_pair_mask     = (int*)    ALIGN64B_MALLOC(INT_SIZE * mat_size);
    TinyDFT->bf_pair_j        = (int*)    ALIGN64B_MALLOC(INT_SIZE * mat_size);
    TinyDFT->bf_pair_diag     = (int*)    ALIGN64B_MALLOC(INT_SIZE * nbf);
    TinyDFT->bf_mask_displs   = (int*)    ALIGN64B_MALLOC(INT_SIZE * (nbf + 1));
    TinyDFT->df_sp_scrval     = (double*) ALIGN64B_MALLOC(DBL_SIZE * df_nshell);
    assert(TinyDFT->df_shell_bf_sind != NULL);
    assert(TinyDFT->df_shell_bf_num  != NULL);
    assert(TinyDFT->bf_pair_mask     != NULL);
    assert(TinyDFT->bf_pair_j        != NULL);
    assert(TinyDFT->bf_pair_diag     != NULL);
    assert(TinyDFT->bf_mask_displs   != NULL);
    assert(TinyDFT->df_sp_scrval     != NULL);
    TinyDFT->mem_size += (double) (INT_SIZE * (df_nshell * 2 + 1));
    TinyDFT->mem_size += (double) (INT_SIZE * (nbf * 2 + 1));
    TinyDFT->mem_size += (double) (INT_SIZE * 2 * mat_size);
    TinyDFT->mem_size += (double) (DBL_SIZE * df_nshell);
    
    for (int i = 0; i < TinyDFT->df_nshell; i++)
    {
        TinyDFT->df_shell_bf_sind[i] = CMS_getFuncStartInd(TinyDFT->df_basis, i);
        TinyDFT->df_shell_bf_num[i]  = CMS_getShellDim    (TinyDFT->df_basis, i);
    }
    TinyDFT->df_shell_bf_sind[TinyDFT->df_nshell] = TinyDFT->df_nbf;
    
    int n_occ     = TinyDFT->n_occ;
    int df_nbf    = TinyDFT->df_nbf;
    int df_nbf_16 = (df_nbf + 15) / 16 * 16;
    TinyDFT->df_nbf_16 = df_nbf_16;
    size_t temp_J_msize = (size_t) df_nbf_16 * (size_t) nthread;
    size_t temp_K_msize = (size_t) df_nbf * (size_t) n_occ * (size_t) nbf;
    size_t df_mat_msize = (size_t) df_nbf * (size_t) df_nbf;
    df_mat_msize *= DBL_SIZE;
    temp_J_msize *= DBL_SIZE;
    temp_K_msize *= DBL_SIZE;
    TinyDFT->Jpq    = (double*) ALIGN64B_MALLOC(df_mat_msize);
    TinyDFT->temp_J = (double*) ALIGN64B_MALLOC(temp_J_msize);
    TinyDFT->temp_K = (double*) ALIGN64B_MALLOC(temp_K_msize);
    assert(TinyDFT->Jpq    != NULL);
    assert(TinyDFT->temp_J != NULL);
    assert(TinyDFT->temp_K != NULL);
    TinyDFT->mem_size += (double) df_mat_msize;
    TinyDFT->mem_size += (double) temp_J_msize;
    TinyDFT->mem_size += (double) temp_K_msize;
    TinyDFT->pqA       = NULL;
    TinyDFT->df_tensor = NULL;
}

static void TinyDFT_init_batch_dgemm(TinyDFT_t TinyDFT)
{
    #define DGEMM_BLK_SIZE 64
    
    int nbf      = TinyDFT->nbf;
    int df_nbf   = TinyDFT->df_nbf;
    int mat_K_BS = nbf / 10;
    if (mat_K_BS < DGEMM_BLK_SIZE) mat_K_BS = DGEMM_BLK_SIZE;
    int nblock   = (nbf + mat_K_BS - 1) / mat_K_BS;
    int nblock0  = nbf / mat_K_BS;
    int bs_rem   = nbf % mat_K_BS;
    int ntile    = (nblock + 1) * nblock / 2;
    TinyDFT->mat_K_ntile = ntile;
    TinyDFT->mat_K_BS    = mat_K_BS;
    TinyDFT->mat_K_group_size = (int*) malloc(sizeof(int) * nbf);
    int *group_size = &TinyDFT->mat_K_group_size[0];
    group_size[0] = (nblock0 * (nblock0 + 1)) / 2;
    if (bs_rem > 0)
    {
        group_size[1] = nblock0;
        group_size[2] = 1;
    } else {
        group_size[1] = 0;
        group_size[2] = 0;
    }
    TinyDFT->mat_K_transa = (CBLAS_TRANSPOSE*) malloc(sizeof(CBLAS_TRANSPOSE) * nbf);
    TinyDFT->mat_K_transb = (CBLAS_TRANSPOSE*) malloc(sizeof(CBLAS_TRANSPOSE) * nbf);
    TinyDFT->mat_K_m      = (int*)             malloc(sizeof(int)             * nbf);
    TinyDFT->mat_K_n      = (int*)             malloc(sizeof(int)             * nbf);
    TinyDFT->mat_K_k      = (int*)             malloc(sizeof(int)             * nbf);
    TinyDFT->mat_K_alpha  = (double*)          malloc(sizeof(double)          * nbf);
    TinyDFT->mat_K_beta   = (double*)          malloc(sizeof(double)          * nbf);
    TinyDFT->mat_K_a      = (double**)         malloc(sizeof(double*)         * nbf);
    TinyDFT->mat_K_b      = (double**)         malloc(sizeof(double*)         * nbf);
    TinyDFT->mat_K_c      = (double**)         malloc(sizeof(double*)         * nbf);
    TinyDFT->mat_K_lda    = (int*)             malloc(sizeof(int)             * nbf);
    TinyDFT->mat_K_ldb    = (int*)             malloc(sizeof(int)             * nbf);
    TinyDFT->mat_K_ldc    = (int*)             malloc(sizeof(int)             * nbf);
    assert(TinyDFT->mat_K_transa != NULL);
    assert(TinyDFT->mat_K_transb != NULL);
    assert(TinyDFT->mat_K_m      != NULL);
    assert(TinyDFT->mat_K_n      != NULL);
    assert(TinyDFT->mat_K_k      != NULL);
    assert(TinyDFT->mat_K_alpha  != NULL);
    assert(TinyDFT->mat_K_beta   != NULL);
    assert(TinyDFT->mat_K_a      != NULL);
    assert(TinyDFT->mat_K_b      != NULL);
    assert(TinyDFT->mat_K_c      != NULL);
    assert(TinyDFT->mat_K_lda    != NULL);
    assert(TinyDFT->mat_K_ldb    != NULL);
    assert(TinyDFT->mat_K_ldc    != NULL);
    TinyDFT->mem_size += (double) (sizeof(CBLAS_TRANSPOSE) * nbf * 2);
    TinyDFT->mem_size += (double) (INT_SIZE * nbf * 7);
    TinyDFT->mem_size += (double) (DBL_SIZE * nbf * 5);
}

static void TinyDFT_prepare_DF_sparsity(TinyDFT_t TinyDFT)
{
    double st = get_wtime_sec();
    
    int    nbf             = TinyDFT->nbf;
    int    df_nbf          = TinyDFT->df_nbf;
    int    mat_size        = TinyDFT->mat_size;
    int    num_total_sp    = TinyDFT->num_total_sp;
    int    num_valid_sp    = TinyDFT->num_valid_sp;
    int    *bf_pair_mask   = TinyDFT->bf_pair_mask;
    int    *bf_pair_j      = TinyDFT->bf_pair_j;
    int    *bf_pair_diag   = TinyDFT->bf_pair_diag;
    int    *bf_mask_displs = TinyDFT->bf_mask_displs;
    double *bf_pair_scrval = TinyDFT->bf_pair_scrval;
    
    // Find the maximum screen value in density fitting shell pairs
    double max_df_scrval = 0;
    for (int i = 0; i < TinyDFT->df_nshell; i++)
    {
        double df_scrval = CMS_Simint_get_DF_sp_scrval(TinyDFT->simint, i);
        TinyDFT->df_sp_scrval[i] = df_scrval;
        if (df_scrval > max_df_scrval) max_df_scrval = df_scrval;
    }
    TinyDFT->max_df_scrval = max_df_scrval;

    // Screen all basis function pairs for DF
    double eta = TinyDFT->shell_scrtol2 / max_df_scrval;
    int bf_pair_nnz = 0;
    bf_mask_displs[0] = 0;
    for (int i = 0; i < nbf; i++)
    {
        int offset_i = i * nbf;
        for (int j = 0; j < nbf; j++)
        {
            if (bf_pair_scrval[offset_i + j] > eta)
            {
                bf_pair_mask[offset_i + j] = bf_pair_nnz;
                bf_pair_j[bf_pair_nnz] = j;
                bf_pair_nnz++;
            } else {
                bf_pair_mask[offset_i + j] = -1;
            }
        }
        // (i, i) always survives screening
        bf_pair_diag[i] = bf_pair_mask[offset_i + i];  
        bf_mask_displs[i + 1] = bf_pair_nnz;
    }
    
    double sp_sparsity = (double) num_valid_sp / (double) num_total_sp;
    double bf_pair_sparsity = (double) bf_pair_nnz / (double) mat_size;
    
    double et = get_wtime_sec();
    double ut = et - st;
    printf("TinyDFT handling shell pair sparsity over,         elapsed time = %.3lf (s)\n", ut);
    
    st = get_wtime_sec();
    size_t tensor_msize = (size_t) bf_pair_nnz * (size_t) df_nbf * DBL_SIZE;
    TinyDFT->pqA       = (double*) ALIGN64B_MALLOC(tensor_msize);
    TinyDFT->df_tensor = (double*) ALIGN64B_MALLOC(tensor_msize);
    assert(TinyDFT->pqA       != NULL);
    assert(TinyDFT->df_tensor != NULL);
    TinyDFT->mem_size += (double) tensor_msize * 2;
    et = get_wtime_sec();
    ut = et - st;
    
    printf("TinyDFT memory allocation and initialization over, elapsed time = %.3lf (s)\n", ut);
    printf("TinyDFT regular + density fitting memory usage = %.2lf MB \n", TinyDFT->mem_size / 1048576.0);
    printf("#### Sparsity of shell / bf pairs = %lf, %lf\n", sp_sparsity, bf_pair_sparsity);
}

static void copy_3center_integral_results(
    int npair, int *P_list, int nint, double *ERIs, int *df_shell_bf_sind, 
    double *pqA, int *bf_pair_mask, int nbf, int df_nbf,
    int startM, int endM, int startN, int endN, int dimN
)
{
    for (int ipair = 0; ipair < npair; ipair++)
    {
        int P = P_list[ipair];
        int startP = df_shell_bf_sind[P];
        int dimP   = df_shell_bf_sind[P + 1] - startP;
        size_t row_mem_size = sizeof(double) * dimP;
        double *ERI_ipair = ERIs + nint * ipair;
        
        for (int iM = startM; iM < endM; iM++)
        {
            int im = iM - startM;
            for (int iN = startN; iN < endN; iN++)
            {
                int in = iN - startN;
                double *eri_ptr = ERI_ipair + (im * dimN + in) * dimP;
                
                int iMN_pair_idx = bf_pair_mask[iM * nbf + iN];
                int iNM_pair_idx = bf_pair_mask[iN * nbf + iM];
                size_t pqA_offset0 = (size_t) iMN_pair_idx * (size_t) df_nbf + (size_t) startP;
                size_t pqA_offset1 = (size_t) iNM_pair_idx * (size_t) df_nbf + (size_t) startP;
                double *pqA_ptr0 = pqA + pqA_offset0;
                double *pqA_ptr1 = pqA + pqA_offset1;
                memcpy(pqA_ptr0, eri_ptr, row_mem_size);
                memcpy(pqA_ptr1, eri_ptr, row_mem_size);
            }
        }
    }
}

static void TinyDFT_calc_DF_3center_int(TinyDFT_t TinyDFT)
{
    int    nbf               = TinyDFT->nbf;
    int    nthread           = TinyDFT->nthread;
    int    df_nbf            = TinyDFT->df_nbf;
    int    nshell            = TinyDFT->nshell;
    int    num_valid_sp      = TinyDFT->num_valid_sp;
    int    df_max_am         = TinyDFT->simint->df_max_am;
    int    *shell_bf_sind    = TinyDFT->shell_bf_sind;
    int    *bf_pair_mask     = TinyDFT->bf_pair_mask;
    int    *valid_sp_lid     = TinyDFT->valid_sp_lid;
    int    *valid_sp_rid     = TinyDFT->valid_sp_rid;
    int    *df_shell_bf_sind = TinyDFT->df_shell_bf_sind;
    int    *df_am_shell_spos = TinyDFT->simint->df_am_shell_spos;
    int    *df_am_shell_id   = TinyDFT->simint->df_am_shell_id;
    double scrtol2           = TinyDFT->shell_scrtol2;
    double *pqA              = TinyDFT->pqA;
    double *sp_scrval        = TinyDFT->sp_scrval;
    double *df_sp_scrval     = TinyDFT->df_sp_scrval;
    Simint_t simint          = TinyDFT->simint;
    
    int *P_lists = (int*) malloc(sizeof(int) * _SIMINT_NSHELL_SIMD * nthread);
    assert(P_lists != NULL);
    
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        int nint, npair;
        int *thread_P_list = P_lists + tid * _SIMINT_NSHELL_SIMD;
        double *thread_ERIs;
        void *multi_sp;
        CMS_Simint_create_multi_sp(&multi_sp);
        
        #pragma omp for schedule(dynamic)
        for (int iMN = 0; iMN < num_valid_sp; iMN++)
        {
            int M = valid_sp_lid[iMN];
            int N = valid_sp_rid[iMN];
            int startM = shell_bf_sind[M];
            int endM   = shell_bf_sind[M + 1];
            int startN = shell_bf_sind[N];
            int endN   = shell_bf_sind[N + 1];
            int dimM   = endM - startM;
            int dimN   = endN - startN;
            double scrval0 = sp_scrval[M * nshell + N];
            
            for (int iAM = 0; iAM <= df_max_am; iAM++)
            {
                npair = 0;
                int iP_start = df_am_shell_spos[iAM];
                int iP_end   = df_am_shell_spos[iAM + 1];
                for (int iP = iP_start; iP < iP_end; iP++)
                {
                    int P = df_am_shell_id[iP];
                    double scrval1 = df_sp_scrval[P];
                    if (scrval0 * scrval1 < scrtol2) continue;
                    
                    thread_P_list[npair] = P;
                    npair++;
                    
                    if (npair == _SIMINT_NSHELL_SIMD)
                    {
                        CMS_Simint_calc_DF_shellquartet_batch(
                            simint, tid, M, N, npair, thread_P_list, 
                            &thread_ERIs, &nint, &multi_sp
                        );
                        if (nint > 0)
                        {
                            copy_3center_integral_results(
                                npair, thread_P_list, nint, thread_ERIs,
                                df_shell_bf_sind, pqA, bf_pair_mask, nbf, df_nbf,
                                startM, endM, startN, endN, dimN
                            );
                        }
                        npair = 0;
                    }
                }  // for (int iP = iP_start; iP < iP_end; iP++)
                
                if (npair > 0)
                {
                    CMS_Simint_calc_DF_shellquartet_batch(
                        simint, tid, M, N, npair, thread_P_list, 
                        &thread_ERIs, &nint, &multi_sp
                    );
                    if (nint > 0)
                    {
                        copy_3center_integral_results(
                            npair, thread_P_list, nint, thread_ERIs,
                            df_shell_bf_sind, pqA, bf_pair_mask, nbf, df_nbf,
                            startM, endM, startN, endN, dimN
                        );
                    }
                    npair = 0;
                } 
            }  // for (int iAM = 0; iAM <= simint->df_max_am; iAM++)
        }  // for (int iMN = 0; iMN < TinyDFT->num_valid_sp; iMN++)
        
        CMS_Simint_free_multi_sp(multi_sp);
    }  // #pragma omp parallel 
    
    free(P_lists);
}

static void TinyDFT_calc_DF_2center_int(TinyDFT_t TinyDFT)
{
    // Fast enough, need not to batch shell quartets
    int    df_nbf            = TinyDFT->df_nbf;
    int    df_nshell         = TinyDFT->df_nshell;
    int    *df_shell_bf_sind = TinyDFT->df_shell_bf_sind;
    double scrtol2           = TinyDFT->shell_scrtol2;
    double *Jpq              = TinyDFT->Jpq;
    double *df_sp_scrval     = TinyDFT->df_sp_scrval;
    Simint_t simint          = TinyDFT->simint;
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nint;
        double *ERIs;
        int flag = 1;
        
        #pragma omp for schedule(dynamic)
        for (int M = 0; M < df_nshell; M++)
        {
            double scrval0 = df_sp_scrval[M];
            for (int N = M; N < df_nshell; N++)
            {
                double scrval1 = df_sp_scrval[N];
                if (scrval0 * scrval1 < scrtol2) continue;

                CMS_Simint_calc_DF_shellpair(simint, tid, M, N, &ERIs, &nint);
                if (nint <= 0) continue;
                
                int startM = df_shell_bf_sind[M];
                int endM   = df_shell_bf_sind[M + 1];
                int startN = df_shell_bf_sind[N];
                int endN   = df_shell_bf_sind[N + 1];
                int dimM   = endM - startM;
                int dimN   = endN - startN;
                
                for (int iM = startM; iM < endM; iM++)
                {
                    int im = iM - startM;
                    for (int iN = startN; iN < endN; iN++)
                    {
                        int in = iN - startN;
                        double I = ERIs[im * dimN + in];
                        Jpq[iM * df_nbf + iN] = I;
                        Jpq[iN * df_nbf + iM] = I;
                    }
                }
            }  // for (int N = i; N < df_nshell; N++)
        }  // for (int M = 0; M < df_nshell; M++)
    }  // #pragma omp parallel
}

static void TinyDFT_calc_invsqrt_Jpq(TinyDFT_t TinyDFT)
{
    int    df_nbf = TinyDFT->df_nbf;
    double *Jpq   = TinyDFT->Jpq;
    
    size_t df_mat_msize = DBL_SIZE * df_nbf * df_nbf;
    double *tmp_mat0  = ALIGN64B_MALLOC(df_mat_msize);
    double *tmp_mat1  = ALIGN64B_MALLOC(df_mat_msize);
    double *df_eigval = ALIGN64B_MALLOC(DBL_SIZE * df_nbf);
    assert(tmp_mat0 != NULL && tmp_mat1 != NULL);
    // Diagonalize Jpq = U * S * U^T, the eigenvectors are stored in tmp_mat0
    memcpy(tmp_mat0, Jpq, df_mat_msize);
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', df_nbf, tmp_mat0, df_nbf, df_eigval);
    // Apply inverse square root to eigen values to get the inverse squart root of Jpq
    for (int i = 0; i < df_nbf; i++)
        df_eigval[i] = 1.0 / sqrt(df_eigval[i]);
    // Right multiply the S^{-1/2} to U
    #pragma omp parallel for
    for (int irow = 0; irow < df_nbf; irow++)
    {
        double *tmp_mat0_ptr = tmp_mat0 + irow * df_nbf;
        double *tmp_mat1_ptr = tmp_mat1 + irow * df_nbf;
        memcpy(tmp_mat1_ptr, tmp_mat0_ptr, DBL_SIZE * df_nbf);
        for (int icol = 0; icol < df_nbf; icol++)
            tmp_mat0_ptr[icol] *= df_eigval[icol];
    }
    // Get Jpq^{-1/2} = U * S^{-1/2} * U', Jpq^{-1/2} is stored in Jpq
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, df_nbf, df_nbf, df_nbf, 
        1.0, tmp_mat0, df_nbf, tmp_mat1, df_nbf, 0.0, Jpq, df_nbf
    );
    ALIGN64B_FREE(tmp_mat0);
    ALIGN64B_FREE(tmp_mat1);
    ALIGN64B_FREE(df_eigval);
}

static void TinyDFT_build_DF_tensor(TinyDFT_t TinyDFT)
{
    double st, et;

    printf("---------- DF tensor construction ----------\n");

    // Calculate 3-center density fitting integrals
    st = get_wtime_sec();
    TinyDFT_calc_DF_3center_int(TinyDFT);
    et = get_wtime_sec();
    printf("* 3-center integral : %.3lf (s)\n", et - st);
    
    // Calculate the Coulomb metric matrix
    st = get_wtime_sec();
    TinyDFT_calc_DF_2center_int(TinyDFT);
    et = get_wtime_sec();
    printf("* 2-center integral : %.3lf (s)\n", et - st);

    // Factorize the Jpq
    st = get_wtime_sec();
    TinyDFT_calc_invsqrt_Jpq(TinyDFT);
    et = get_wtime_sec();
    printf("* matrix inv-sqrt   : %.3lf (s)\n", et - st);

    // Form the density fitting tensor
    st = get_wtime_sec();
    int    nbf         = TinyDFT->nbf;
    int    df_nbf      = TinyDFT->df_nbf;
    int    bf_pair_cnt = TinyDFT->bf_mask_displs[nbf];
    double *Jpq        = TinyDFT->Jpq;
    double *pqA        = TinyDFT->pqA;
    double *df_tensor  = TinyDFT->df_tensor;
    // df_tensor(i, j, k) = dot(pqA(i, j, 1:df_nbf), Jpq_invsqrt(1:df_nbf, k))
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, bf_pair_cnt, df_nbf, df_nbf,
        1.0, pqA, df_nbf, Jpq, df_nbf, 0.0, df_tensor, df_nbf
    );
    et = get_wtime_sec();
    printf("* build DF tensor   : %.3lf (s)\n", et - st);

    printf("---------- DF tensor construction finished ----------\n");
}

// Set up density fitting
void TinyDFT_setup_DF(TinyDFT_t TinyDFT, char *df_bas_fname, char *xyz_fname)
{
    assert(TinyDFT != NULL);
    
    TinyDFT_load_DF_basis(TinyDFT, df_bas_fname, xyz_fname);
    
    TinyDFT_init_batch_dgemm(TinyDFT);
    
    TinyDFT_prepare_DF_sparsity(TinyDFT);
 
    TinyDFT_build_DF_tensor(TinyDFT);
    
    CMS_Simint_free_DF_shellpairs(TinyDFT->simint);
}
