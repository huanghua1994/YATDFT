#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#include "linalg_lib_wrapper.h"

#include "utils.h"
#include "TinyDFT_typedef.h"
#include "build_JKmat_DF.h"

void TinyDFT_reduce_temp_J(double *temp_J, double *temp_J_thread, int len, int tid, int nthread)
{
    while (nthread > 1)
    {
        int mid = (nthread + 1) / 2;
        int act_mid = nthread / 2;
        if (tid < act_mid)
        {
            double *dst = temp_J_thread + len * mid;

            #pragma omp simd
            for (int i = 0; i < len; i++)
                temp_J_thread[i] += dst[i];
        }

        #pragma omp barrier
        nthread = mid;
    }
}

// Build temporary array for J matrix and form J matrix
// Low flop-per-byte ratio: access: nbf^2 * (df_nbf+1), compute: nbf^2 * df_nbf 
// Note: the J_mat is not completed, the symmetrizing is done later
static void TinyDFT_build_Jmat_DF(TinyDFT_t TinyDFT, const double *D_mat, double *J_mat, double *temp_J_t, double *J_mat_t)
{
    int    nbf             = TinyDFT->nbf;
    int    df_nbf          = TinyDFT->df_nbf;
    int    df_nbf_16       = TinyDFT->df_nbf_16;
    int    nthread         = TinyDFT->nthread;
    int    *bf_pair_j      = TinyDFT->bf_pair_j;
    int    *bf_pair_diag   = TinyDFT->bf_pair_diag;
    int    *bf_mask_displs = TinyDFT->bf_mask_displs;
    double *temp_J         = TinyDFT->temp_J;
    double *df_tensor      = TinyDFT->df_tensor;
    
    double t0, t1, t2;
    #pragma omp parallel
    {
        int tid  = omp_get_thread_num();

        #pragma omp master
        t0 = get_wtime_sec();
        
        // Use thread local buffer (aligned to 128B) to reduce false sharing
        double *temp_J_thread = temp_J + df_nbf_16 * tid;
        
        // Generate temporary array for J
        memset(temp_J_thread, 0, sizeof(double) * df_nbf);
        
        #pragma omp for schedule(dynamic)
        for (int k = 0; k < nbf; k++)
        {
            int diag_k_idx = bf_pair_diag[k];
            int idx_kk = k * nbf + k;
            
            // Basis function pair (i, i) always survives screening
            size_t offset = (size_t) diag_k_idx * (size_t) df_nbf;
            double *df_tensor_row = df_tensor + offset;
            double D_kl = D_mat[idx_kk];
            #pragma omp simd
            for (size_t p = 0; p < df_nbf; p++)
                temp_J_thread[p] += D_kl * df_tensor_row[p];
            
            
            int row_k_epos = bf_mask_displs[k + 1];
            for (int l_idx = diag_k_idx + 1; l_idx < row_k_epos; l_idx++)
            {
                int l = bf_pair_j[l_idx];
                int idx_kl = k * nbf + l;
                double D_kl = D_mat[idx_kl] * 2.0;
                size_t offset = (size_t) l_idx * (size_t) df_nbf;
                double *df_tensor_row = df_tensor + offset;
                
                #pragma omp simd
                for (size_t p = 0; p < df_nbf; p++)
                    temp_J_thread[p] += D_kl * df_tensor_row[p];
            }
        }
        
        #pragma omp barrier
        TinyDFT_reduce_temp_J(temp_J, temp_J_thread, df_nbf_16, tid, nthread);
        
        #pragma omp master
        t1 = get_wtime_sec();

        // Build J matrix
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < nbf; i++)
        {
            int diag_i_idx = bf_pair_diag[i];
            int row_i_epos = bf_mask_displs[i + 1];
            for (int j_idx = diag_i_idx; j_idx < row_i_epos; j_idx++)
            {
                int j = bf_pair_j[j_idx];
                
                size_t offset = (size_t) j_idx * (size_t) df_nbf;
                double *df_tensor_row = df_tensor + offset;
                
                double t = 0;
                #pragma omp simd
                for (size_t p = 0; p < df_nbf; p++)
                    t += temp_J[p] * df_tensor_row[p];
                J_mat[i * nbf + j] = t;
            }
        }
        
        #pragma omp master
        t2 = get_wtime_sec();
    }
    
    *temp_J_t = t1 - t0;
    *J_mat_t  = t2 - t1;
}

static void TinyDFT_set_batch_dgemm_temp_K(TinyDFT_t TinyDFT)
{
    int    nbf             = TinyDFT->nbf;
    int    df_nbf          = TinyDFT->df_nbf;
    int    n_occ           = TinyDFT->n_occ;
    int    *bf_mask_displs = TinyDFT->bf_mask_displs;
    double *Cocc_tmp       = TinyDFT->pqA;
    double *df_tensor      = TinyDFT->df_tensor;
    double *temp_K         = TinyDFT->temp_K;
    
    for (int i = 0; i < nbf; i++)
    {
        int row_spos = bf_mask_displs[i];
        int row_epos = bf_mask_displs[i + 1];
        int row_len  = row_epos - row_spos;
        
        size_t offset_a = (size_t) row_spos * (size_t) df_nbf;
        size_t offset_b = (size_t) row_spos * (size_t) df_nbf;
        size_t offset_c = (size_t) i * (size_t) n_occ * (size_t) df_nbf;
        double *A_ptr = Cocc_tmp  + offset_a;
        double *B_ptr = df_tensor + offset_b;
        double *C_ptr = temp_K    + offset_c;
        
        TinyDFT->mat_K_transa[i] = CblasTrans;
        TinyDFT->mat_K_transb[i] = CblasNoTrans;
        TinyDFT->mat_K_m[i]      = n_occ;
        TinyDFT->mat_K_n[i]      = df_nbf;
        TinyDFT->mat_K_k[i]      = row_len;
        TinyDFT->mat_K_alpha[i]  = 1.0;
        TinyDFT->mat_K_beta[i]   = 0.0;
        TinyDFT->mat_K_a[i]      = A_ptr;
        TinyDFT->mat_K_b[i]      = B_ptr;
        TinyDFT->mat_K_c[i]      = C_ptr;
        TinyDFT->mat_K_lda[i]    = df_nbf;
        TinyDFT->mat_K_ldb[i]    = df_nbf;
        TinyDFT->mat_K_ldc[i]    = df_nbf;
        TinyDFT->mat_K_group_size[i] = 1;
    }
}

static void TinyDFT_set_batch_dgemm_K(TinyDFT_t TinyDFT, double *K_mat)
{
    int nbf      = TinyDFT->nbf;
    int df_nbf   = TinyDFT->df_nbf;
    int n_occ    = TinyDFT->n_occ;
    int mat_K_BS = TinyDFT->mat_K_BS;
    int nblock0  = nbf / mat_K_BS;
    int bs_rem   = nbf % mat_K_BS;
    int *group_size = TinyDFT->mat_K_group_size;
    group_size[0] = (nblock0 * (nblock0 + 1)) / 2;
    if (bs_rem > 0)
    {
        group_size[1] = nblock0;
        group_size[2] = 1;
    } else {
        group_size[1] = 0;
        group_size[2] = 0;
    }
    
    double *temp_K = TinyDFT->temp_K;
    int cnt0 = 0, cnt1 = group_size[0];
    int cnt2 = group_size[0] + group_size[1];
    for (int i = 0; i < nbf; i += mat_K_BS)
    {
        int i_len = mat_K_BS < (nbf - i) ? mat_K_BS : (nbf - i);
        for (int j = i; j < nbf; j += mat_K_BS)
        {
            int j_len = mat_K_BS < (nbf - j) ? mat_K_BS : (nbf - j);
            
            size_t offset_i0 = (size_t) i * (size_t) n_occ * (size_t) df_nbf;
            size_t offset_j0 = (size_t) j * (size_t) n_occ * (size_t) df_nbf;
            double *K_ij     = K_mat  + i * nbf + j;
            double *temp_K_i = temp_K + offset_i0;
            double *temp_K_j = temp_K + offset_j0;
            
            int cnt, gid;
            if ((i_len == mat_K_BS) && (j_len == mat_K_BS))
            {
                cnt = cnt0;
                gid = 0;
                cnt0++;
            } else {
                if ((i_len == mat_K_BS) && (j_len < mat_K_BS)) 
                {
                    cnt = cnt1;
                    gid = 1;
                    cnt1++;
                } else {
                    cnt = cnt2;
                    gid = 2;
                }
            }
            
            TinyDFT->mat_K_transa[gid] = CblasNoTrans;
            TinyDFT->mat_K_transb[gid] = CblasTrans;
            TinyDFT->mat_K_m[gid]      = i_len;
            TinyDFT->mat_K_n[gid]      = j_len;
            TinyDFT->mat_K_k[gid]      = n_occ * df_nbf;
            TinyDFT->mat_K_alpha[gid]  = 1.0;
            TinyDFT->mat_K_beta[gid]   = 0.0;
            TinyDFT->mat_K_a[cnt]      = temp_K_i;
            TinyDFT->mat_K_b[cnt]      = temp_K_j;
            TinyDFT->mat_K_c[cnt]      = K_ij;
            TinyDFT->mat_K_lda[gid]    = n_occ * df_nbf;
            TinyDFT->mat_K_ldb[gid]    = n_occ * df_nbf;
            TinyDFT->mat_K_ldc[gid]    = nbf;
        }
    }
}

#ifndef USE_MKL
#warning cblas_dgemm_batch() is not available in your BLAS library, will use cblas_dgemm to simulate it. 
void cblas_dgemm_batch(
    const CBLAS_LAYOUT Layout, 
    const CBLAS_TRANSPOSE *transa_array, 
    const CBLAS_TRANSPOSE *transb_array, 
    const int *m_array, const int *n_array, const int *k_array, 
    const double *alpha_array, 
    const double **a_array, const int *lda_array, 
    const double **b_array, const int *ldb_array, 
    const double *beta_array, 
    double **c_array, const int *ldc_array, 
    const int group_count, const int *group_size
)
{
    int idx = 0;
    for (int i = 0; i < group_count; i++)
    {
        const CBLAS_TRANSPOSE transa_i = transa_array[i];
        const CBLAS_TRANSPOSE transb_i = transb_array[i];
        const int m_i = m_array[i];
        const int n_i = n_array[i];
        const int k_i = k_array[i];
        const int lda_i = lda_array[i];
        const int ldb_i = ldb_array[i];
        const int ldc_i = ldc_array[i];
        const double alpha_i = alpha_array[i];
        const double beta_i  = beta_array[i];
        for (int j = 0; j < group_size[i]; j++)
        {
            const double *a_idx = a_array[idx + j];
            const double *b_idx = b_array[idx + j];
            double *c_idx = c_array[idx + j];
            cblas_dgemm(
                Layout, transa_i, transb_i, m_i, n_i, k_i,
                alpha_i, a_idx, lda_i, b_idx, ldb_i, beta_i, c_idx, ldc_i
            );
        }
        idx += group_size[i];
    }
}
#endif

// Build temporary tensor for K matrix and form K matrix using Cocc matrix
// High flop-per-byte ratio: access: nbf * df_nbf * (nbf + n_occ) , compute: nbf^2 * df_nbf * n_occ
// Note: the K_mat is not completed, the symmetrizing is done later
static void TinyDFT_build_Kmat_DF(TinyDFT_t TinyDFT, const double *Cocc_mat, double *K_mat, double *temp_K_t, double *K_mat_t)
{
    int    nbf             = TinyDFT->nbf;
    int    df_nbf          = TinyDFT->df_nbf;
    int    df_save_mem     = TinyDFT->df_save_mem;
    int    n_occ           = TinyDFT->n_occ;
    int    ngroups_temp_K  = nbf;
    int    bf_pair_cnt     = TinyDFT->bf_mask_displs[nbf];
    int    *bf_pair_j      = TinyDFT->bf_pair_j;
    int    *bf_mask_displs = TinyDFT->bf_mask_displs;
    double *df_tensor      = TinyDFT->df_tensor;
    double *temp_K         = TinyDFT->temp_K;
    double *Cocc_tmp       = TinyDFT->pqA;
    
    double t0, t1, t2;
    
    // Construct temporary tensor for K matrix
    // Formula: temp_K(i, s, p) = dot(Cocc_mat(1:nbf, s), df_tensor(i, 1:nbf, p))
    t0 = get_wtime_sec();
    if (df_save_mem == 0)
    {
        TinyDFT_set_batch_dgemm_temp_K(TinyDFT);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < bf_pair_cnt; i++)
        {
            int j = bf_pair_j[i];
            size_t Cocc_tmp_offset = (size_t) i * (size_t) df_nbf;
            size_t Cocc_mat_offset = (size_t) j * (size_t) n_occ;
            double *Cocc_tmp_ptr = Cocc_tmp + Cocc_tmp_offset;
            const double *Cocc_mat_ptr = Cocc_mat + Cocc_mat_offset;
            memcpy(Cocc_tmp_ptr, Cocc_mat_ptr, DBL_MSIZE * n_occ);
        }
        cblas_dgemm_batch(
            CblasRowMajor, TinyDFT->mat_K_transa, TinyDFT->mat_K_transb,
            TinyDFT->mat_K_m, TinyDFT->mat_K_n, TinyDFT->mat_K_k,
            TinyDFT->mat_K_alpha,  
            (const double **) TinyDFT->mat_K_a, TinyDFT->mat_K_lda,
            (const double **) TinyDFT->mat_K_b, TinyDFT->mat_K_ldb,
            TinyDFT->mat_K_beta,
            TinyDFT->mat_K_c, TinyDFT->mat_K_ldc,
            ngroups_temp_K, TinyDFT->mat_K_group_size
        );
    } else {
        double *A_ptr  = TinyDFT->Cocc_mat;
        double *temp_A = TinyDFT->tmp_mat;
        for (int i = 0; i < nbf; i++)
        {
            size_t offset_c = (size_t) i * (size_t) n_occ * (size_t) df_nbf;
            double *C_ptr = temp_K + offset_c;
            
            int j_idx_spos = bf_mask_displs[i];
            int j_idx_epos = bf_mask_displs[i + 1];
            for (int j_idx = j_idx_spos; j_idx < j_idx_epos; j_idx++)
            {
                int j = bf_pair_j[j_idx];
                int cnt = j_idx - j_idx_spos;
                memcpy(temp_A + cnt * n_occ,  A_ptr + j * n_occ,  DBL_MSIZE * n_occ);
            }
            
            int ncols = j_idx_epos - j_idx_spos;
            size_t df_tensor_offset = (size_t) j_idx_spos * (size_t) df_nbf;
            double *df_tensor_ptr = df_tensor + df_tensor_offset;
            cblas_dgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans, n_occ, df_nbf, ncols,
                1.0, temp_A, n_occ, df_tensor_ptr, df_nbf, 0.0, C_ptr, df_nbf
            );
        }
    }  // End of "if (df_save_mem == 0)"
    t1 = get_wtime_sec();

    // Build K matrix
    // Formula: K(i, j) = sum_{s=1}^{n_occ} [ dot(temp_K(i, s, 1:df_nbf), temp_K(j, s, 1:df_nbf)) ]
    if (nbf <= 1024)
    {    
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans, nbf, nbf, n_occ * df_nbf, 
            1.0, temp_K, n_occ * df_nbf, temp_K, n_occ * df_nbf, 0.0, K_mat, nbf
        );
    } else {
        int ngroups = 3;
        TinyDFT_set_batch_dgemm_K(TinyDFT, K_mat);
        if (TinyDFT->mat_K_group_size[1] == 0) ngroups = 1;
        cblas_dgemm_batch(
            CblasRowMajor, TinyDFT->mat_K_transa, TinyDFT->mat_K_transb,
            TinyDFT->mat_K_m, TinyDFT->mat_K_n, TinyDFT->mat_K_k,
            TinyDFT->mat_K_alpha,  
            (const double **) TinyDFT->mat_K_a, TinyDFT->mat_K_lda,
            (const double **) TinyDFT->mat_K_b, TinyDFT->mat_K_ldb,
            TinyDFT->mat_K_beta,
            TinyDFT->mat_K_c, TinyDFT->mat_K_ldc,
            ngroups, TinyDFT->mat_K_group_size
        );
    }
    
    t2 = get_wtime_sec();
    
    *temp_K_t = t1 - t0;
    *K_mat_t  = t2 - t1;
}

void TinyDFT_build_JKmat_DF(TinyDFT_t TinyDFT, const double *D_mat, const double *Cocc_mat, double *J_mat, double *K_mat)
{
    if (J_mat == NULL && K_mat == NULL) return;
    
    int nbf = TinyDFT->nbf;
    
    double st, et, total_t, symm_t;
    double temp_J_t = 0.0, J_mat_t = 0.0;
    double temp_K_t = 0.0, K_mat_t = 0.0;

    if (J_mat != NULL)
    {
        TinyDFT_build_Jmat_DF(TinyDFT, D_mat, J_mat, &temp_J_t, &J_mat_t);
        st = get_wtime_sec();
        #pragma omp for schedule(dynamic)
        for (int i = 1; i < nbf; i++)
        {
            #pragma omp simd
            for (int j = 0; j < i; j++)
                J_mat[i * nbf + j] = J_mat[j * nbf + i];
        }
        et = get_wtime_sec();
        symm_t  = et - st;
        total_t = temp_J_t + J_mat_t + symm_t;
        printf(
            "* Build J mat using DF  : %.3lf (s), "
            "aux / Jmat / symm = %.3lf, %.3lf, %.3lf\n", 
            total_t, temp_J_t, J_mat_t, symm_t
        );
    }
    
    if (K_mat != NULL)
    {
        if (TinyDFT->temp_K == NULL)
        {
            size_t temp_K_msize = (size_t) TinyDFT->df_nbf * (size_t) TinyDFT->n_occ * (size_t) TinyDFT->nbf;
            temp_K_msize *= DBL_MSIZE;
            st = get_wtime_sec();
            TinyDFT->temp_K = (double*) malloc_aligned(temp_K_msize, 64);
            assert(TinyDFT->temp_K != NULL);
            et = get_wtime_sec();
            TinyDFT->mem_size += (double) temp_K_msize;
            printf("Allocate auxiliary tensor for density fitting K matrix build : %.3lf (s)\n", et - st);
        }
        
        TinyDFT_build_Kmat_DF(TinyDFT, Cocc_mat, K_mat, &temp_K_t, &K_mat_t);
        st = get_wtime_sec();
        #pragma omp for schedule(dynamic)
        for (int i = 1; i < nbf; i++)
        {
            #pragma omp simd
            for (int j = 0; j < i; j++)
                K_mat[i * nbf + j] = K_mat[j * nbf + i];
        }
        et = get_wtime_sec();
        symm_t  = et - st;
        total_t = temp_K_t + K_mat_t + symm_t;
        printf(
            "* Build K mat using DF  : %.3lf (s), "
            "aux / Kmat / symm = %.3lf, %.3lf, %.3lf\n", 
            total_t, temp_K_t, K_mat_t, symm_t
        );
    }
}

