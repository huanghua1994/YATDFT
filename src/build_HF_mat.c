#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "linalg_lib_wrapper.h"

#include "utils.h"
#include "TinyDFT_typedef.h"
#include "build_HF_mat.h"
#include "ket_sp_list.h"
#include "acc_JKmat.h"
#include "libCMS.h"

void TinyDFT_build_Hcore_S_X_mat(TinyDFT_t TinyDFT, double *Hcore_mat, double *S_mat, double *X_mat)
{
    assert(TinyDFT != NULL);
    
    int nbf            = TinyDFT->nbf;
    int nshell         = TinyDFT->nshell;
    int mat_size       = TinyDFT->mat_size;
    int *shell_bf_sind = TinyDFT->shell_bf_sind;
    int *shell_bf_num  = TinyDFT->shell_bf_num;
    Simint_t   simint  = TinyDFT->simint;
    BasisSet_t basis   = TinyDFT->basis;
    
    // Compute core Hamiltonian and overlap matrix
    memset(Hcore_mat, 0, DBL_MSIZE * mat_size);
    memset(S_mat,     0, DBL_MSIZE * mat_size);
    #pragma omp parallel for schedule(dynamic)
    for (int M = 0; M < nshell; M++)
    {
        int tid = omp_get_thread_num();
        for (int N = 0; N < nshell; N++)
        {
            int nint, offset, nrows, ncols;
            double *integrals, *S_ptr, *Hcore_ptr;
            
            offset    = shell_bf_sind[M] * nbf + shell_bf_sind[N];
            S_ptr     = S_mat  + offset;
            Hcore_ptr = Hcore_mat + offset;
            nrows     = shell_bf_num[M];
            ncols     = shell_bf_num[N];
            
            // Compute the contribution of current shell pair to core Hamiltonian matrix
            CMS_Simint_calc_pair_ovlp(simint, tid, M, N, &integrals, &nint);
            if (nint > 0) copy_dbl_mat_blk(integrals, ncols, nrows, ncols, S_ptr, nbf);
            
            // Compute the contribution of current shell pair to overlap matrix
            CMS_Simint_calc_pair_Hcore(basis, simint, tid, M, N, &integrals, &nint);
            if (nint > 0) copy_dbl_mat_blk(integrals, ncols, nrows, ncols, Hcore_ptr, nbf);
        }
    }
    
    // Construct basis transformation matrix
    double *workbuf = (double*) malloc(sizeof(double) * (2 * mat_size + nbf));
    assert(workbuf != NULL);
    double *U_mat  = workbuf; 
    double *U0_mat = U_mat  + mat_size;
    double *eigval = U0_mat + mat_size;
    // [U, D] = eig(S);
    memcpy(U_mat, S_mat, DBL_MSIZE * mat_size);
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', nbf, U_mat, nbf, eigval); // U_mat will be overwritten by eigenvectors
    // X = U * D^{-1/2} * U'^T
    memcpy(U0_mat, U_mat, DBL_MSIZE * mat_size);
    int cnt = 0;
    double S_ev_thres = 1.0e-6;
    for (int i = 0; i < nbf; i++) 
    {
        if (eigval[i] >= eigval[nbf - 1] * S_ev_thres)
        {
            eigval[i] = 1.0 / sqrt(eigval[i]);
            cnt++;
        } else {
            eigval[i] = 0.0;
        }
    }
    printf("Overlap matrix S truncation: reltol = %.2e, %d out of %d eigenvectors are preserved\n", S_ev_thres, cnt, nbf);
    for (int i = 0; i < nbf; i++)
    {
        #pragma omp simd
        for (int j = 0; j < nbf; j++)
            U0_mat[i * nbf + j] *= eigval[j];
    }
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, nbf, nbf, nbf, 
        1.0, U0_mat, nbf, U_mat, nbf, 0.0, X_mat, nbf
    );
    free(workbuf);
}

// Get the final J and K matrices: J = (J + J^T) / 2, K = (K + K^T) / 2
static void TinyDFT_finalize_JKmat(const int nbf, double *J_mat, double *K_mat, const int build_J, const int build_K)
{
    if (build_J == 1 && build_K == 1)
    {
        #pragma omp for schedule(dynamic)
        for (int irow = 0; irow < nbf; irow++)
        {
            for (int icol = irow + 1; icol < nbf; icol++)
            {
                int idx1 = irow * nbf + icol;
                int idx2 = icol * nbf + irow;
                double Jval = (J_mat[idx1] + J_mat[idx2]) * 0.5;
                double Kval = (K_mat[idx1] + K_mat[idx2]) * 0.5;
                J_mat[idx1] = Jval;
                J_mat[idx2] = Jval;
                K_mat[idx1] = Kval;
                K_mat[idx2] = Kval;
            }
        }
    }
    
    if (build_J == 1 && build_K == 0)
    {
        #pragma omp for schedule(dynamic)
        for (int irow = 0; irow < nbf; irow++)
        {
            for (int icol = irow + 1; icol < nbf; icol++)
            {
                int idx1 = irow * nbf + icol;
                int idx2 = icol * nbf + irow;
                double Jval = (J_mat[idx1] + J_mat[idx2]) * 0.5;
                J_mat[idx1] = Jval;
                J_mat[idx2] = Jval;
            }
        }
    }
    
    if (build_J == 0 && build_K == 1)
    {
        #pragma omp for schedule(dynamic)
        for (int irow = 0; irow < nbf; irow++)
        {
            for (int icol = irow + 1; icol < nbf; icol++)
            {
                int idx1 = irow * nbf + icol;
                int idx2 = icol * nbf + irow;
                double Kval = (K_mat[idx1] + K_mat[idx2]) * 0.5;
                K_mat[idx1] = Kval;
                K_mat[idx2] = Kval;
            }
        }
    }
}

static void TinyDFT_JKblkmat_to_JKmat(
    const int nshell, const int nbf, const int *shell_bf_num, const int *shell_bf_sind, 
    const int *blk_mat_ptr, const double *J_blk_mat, const double *K_blk_mat, 
    double *J_mat, double *K_mat, const int build_J, const int build_K
)
{
    if (build_J)
    {
        #pragma omp for
        for (int i = 0; i < nshell; i++)
        {
            for (int j = 0; j < nshell; j++)
            {
                int Jblk_offset = blk_mat_ptr[i * nshell + j];
                int J_offset    = shell_bf_sind[i] * nbf + shell_bf_sind[j];
                copy_dbl_mat_blk(
                    J_blk_mat + Jblk_offset, shell_bf_num[j],
                    shell_bf_num[i], shell_bf_num[j],
                    J_mat + J_offset, nbf
                );
            }
        }
    }
    
    if (build_K)
    {
        #pragma omp for
        for (int i = 0; i < nshell; i++)
        {
            for (int j = 0; j < nshell; j++)
            {
                int Kblk_offset = blk_mat_ptr[i * nshell + j];
                int K_offset    = shell_bf_sind[i] * nbf + shell_bf_sind[j];
                copy_dbl_mat_blk(
                    K_blk_mat + Kblk_offset, shell_bf_num[j],
                    shell_bf_num[i], shell_bf_num[j],
                    K_mat + K_offset, nbf
                );
            }
        }
    }
}

static void TinyDFT_Dmat_to_Dblkmat(
    const int nshell, const int nbf, const int *shell_bf_num, const int *shell_bf_sind, 
    const int *blk_mat_ptr, const double *D_mat, double *D_blk_mat
)
{
    #pragma omp for
    for (int i = 0; i < nshell; i++)
    {
        for (int j = 0; j < nshell; j++)
        {
            int Dblk_offset = blk_mat_ptr[i * nshell + j];
            int D_offset    = shell_bf_sind[i] * nbf + shell_bf_sind[j];
            copy_dbl_mat_blk(
                D_mat + D_offset, nbf, 
                shell_bf_num[i], shell_bf_num[j],
                D_blk_mat + Dblk_offset, shell_bf_num[j]
            );
        }
    }
}

void TinyDFT_build_JKmat(TinyDFT_t TinyDFT, const double *D_mat, double *J_mat, double *K_mat)
{
    int    nbf            = TinyDFT->nbf;
    int    nshell         = TinyDFT->nshell;
    int    num_valid_sp   = TinyDFT->num_valid_sp;
    int    max_dim        = TinyDFT->max_dim;
    int    mat_size       = TinyDFT->mat_size;
    int    max_JKacc_buf  = TinyDFT->max_JKacc_buf;
    int    *shell_bf_num  = TinyDFT->shell_bf_num;
    int    *shell_bf_sind = TinyDFT->shell_bf_sind;
    int    *valid_sp_lid  = TinyDFT->valid_sp_lid;
    int    *valid_sp_rid  = TinyDFT->valid_sp_rid;
    int    *blk_mat_ptr   = TinyDFT->blk_mat_ptr;
    int    *Mpair_flag    = TinyDFT->Mpair_flag;
    int    *Npair_flag    = TinyDFT->Npair_flag;
    double scrtol2        = TinyDFT->shell_scrtol2;
    double *sp_scrval     = TinyDFT->sp_scrval;
    double *J_blk_mat     = TinyDFT->J_blk_mat;
    double *K_blk_mat     = TinyDFT->K_blk_mat;
    double *D_blk_mat     = TinyDFT->D_blk_mat;
    double *FM_strip_buf  = TinyDFT->FM_strip_buf;
    double *FN_strip_buf  = TinyDFT->FN_strip_buf;
    Simint_t simint       = TinyDFT->simint;
    
    int build_J = (J_mat == NULL) ? 0 : 1;
    int build_K = (K_mat == NULL) ? 0 : 1;
    if (build_J == 0 && build_K == 0) return;
    
    if (build_J) memset(J_blk_mat, 0, DBL_MSIZE * mat_size);
    if (build_K) memset(K_blk_mat, 0, DBL_MSIZE * mat_size);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        TinyDFT_Dmat_to_Dblkmat(
            nshell, nbf, shell_bf_num, shell_bf_sind, 
            blk_mat_ptr, D_mat, D_blk_mat
        );
        
        // Create ERI batching auxiliary data structures
        // Ket-side shell pair lists that needs to be computed
        ThreadKetShellpairLists_t thread_ksp_lists;
        create_ThreadKetShellpairLists(&thread_ksp_lists);
        // Simint multi_shellpair buffer for batched ERI computation
        void *thread_multi_shellpair;
        CMS_Simint_create_multi_sp(&thread_multi_shellpair);
        
        double *thread_FM_strip_buf = FM_strip_buf + tid * nbf * max_dim;
        double *thread_FN_strip_buf = FN_strip_buf + tid * nbf * max_dim;
        int    *thread_Mpair_flag   = Mpair_flag   + tid * nshell;
        int    *thread_Npair_flag   = Npair_flag   + tid * nshell;
        
        #pragma omp for schedule(dynamic)
        for (int MN = 0; MN < num_valid_sp; MN++)
        {
            int M = valid_sp_lid[MN];
            int N = valid_sp_rid[MN];
            double scrval1 = sp_scrval[M * nshell + N];
            
            double *J_MN_buf = TinyDFT->JKacc_buf + tid * max_JKacc_buf;
            double *J_MN = J_blk_mat + blk_mat_ptr[M * nshell + N];
            int dimM = shell_bf_num[M], dimN = shell_bf_num[N];
            
            if (build_J) memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
            if (build_K)
            {                
                memset(thread_FM_strip_buf, 0, sizeof(double) * nbf * max_dim);
                memset(thread_FN_strip_buf, 0, sizeof(double) * nbf * max_dim);
                memset(thread_Mpair_flag,   0, sizeof(int)    * nshell);
                memset(thread_Npair_flag,   0, sizeof(int)    * nshell);
            }
            
            for (int PQ = 0; PQ < num_valid_sp; PQ++)
            {
                int P = valid_sp_lid[PQ];
                int Q = valid_sp_rid[PQ];
                double scrval2 = sp_scrval[P * nshell + Q];
                
                // Symmetric uniqueness check, from GTFock
                if ((M > P && (M + P) % 2 == 1) || 
                    (M < P && (M + P) % 2 == 0))
                continue; 
                
                if ((M == P) &&    ((N > Q && (N + Q) % 2 == 1) ||
                    (N < Q && (N + Q) % 2 == 0)))
                continue;
                
                // Shell screening 
                if (fabs(scrval1 * scrval2) <= scrtol2) continue;
                
                // Push ket-side shell pair to corresponding list
                int ket_id = CMS_Simint_get_sp_AM_idx(simint, P, Q);
                KetShellpairList_t dst_sp_list = &thread_ksp_lists->ket_shellpair_lists[ket_id];
                add_shellpair_to_KetShellPairList(dst_sp_list, P, Q);
                
                // If the ket-side shell pair list we just used is full, handle it
                if (dst_sp_list->npairs == MAX_LIST_SIZE)
                {
                    double *thread_batch_eris;
                    int thread_nints;
                    
                    // Compute batched ERIs
                    CMS_Simint_calc_shellquartet_batch(
                        simint, tid, M, N,
                        dst_sp_list->npairs,
                        dst_sp_list->P_list,
                        dst_sp_list->Q_list,
                        &thread_batch_eris, &thread_nints, 
                        &thread_multi_shellpair
                    );
                    
                    // Accumulate ERI results to global matrices
                    if (thread_nints > 0)
                    {
                        double st = get_wtime_sec();
                        acc_JKmat_with_ket_sp_list(
                            TinyDFT, tid, M, N, 
                            dst_sp_list->npairs,
                            dst_sp_list->P_list,
                            dst_sp_list->Q_list,
                            thread_batch_eris,   thread_nints,
                            thread_FM_strip_buf, thread_FN_strip_buf,
                            thread_Mpair_flag,   thread_Npair_flag,
                            build_J, build_K
                        );
                        double et = get_wtime_sec();
                        if (tid == 0) CMS_Simint_add_accF_timer(simint, et - st);
                    }
                    
                    // Reset the computed ket-side shell pair list
                    dst_sp_list->npairs = 0;
                }  // End of "if (dst_sp_list->npairs == MAX_LIST_SIZE)"
            }  // End of PQ loop
            
            // Handles all non-empty ket-side shell pair lists
            for (int ket_id = 0; ket_id < MAX_AM_PAIRS; ket_id++)
            {
                KetShellpairList_t dst_sp_list = &thread_ksp_lists->ket_shellpair_lists[ket_id];
                
                if (dst_sp_list->npairs > 0)
                {
                    double *thread_batch_eris;
                    int thread_nints;
                    
                    // Compute batched ERIs
                    CMS_Simint_calc_shellquartet_batch(
                        simint, tid, M, N,
                        dst_sp_list->npairs,
                        dst_sp_list->P_list,
                        dst_sp_list->Q_list,
                        &thread_batch_eris, &thread_nints, 
                        &thread_multi_shellpair
                    );
                    
                    // Accumulate ERI results to global matrices
                    if (thread_nints > 0)
                    {
                        double st = get_wtime_sec();
                        acc_JKmat_with_ket_sp_list(
                            TinyDFT, tid, M, N, 
                            dst_sp_list->npairs,
                            dst_sp_list->P_list,
                            dst_sp_list->Q_list,
                            thread_batch_eris,   thread_nints,
                            thread_FM_strip_buf, thread_FN_strip_buf,
                            thread_Mpair_flag,   thread_Npair_flag,
                            build_J, build_K
                        );
                        double et = get_wtime_sec();
                        if (tid == 0) CMS_Simint_add_accF_timer(simint, et - st);
                    }
                    
                    // Reset the computed ket-side shell pair list
                    dst_sp_list->npairs = 0;
                }  // End of "if (dst_sp_list->npairs > 0)"
            }  // End of ket_id loop
            
            // Accumulate thread-local J and K results to global J and K mat
            if (build_J) atomic_add_vector(J_MN, J_MN_buf, dimM * dimN);
            if (build_K) 
            {                
                int FM_strip_offset = blk_mat_ptr[M * nshell];
                int FN_strip_offset = blk_mat_ptr[N * nshell];
                for (int iPQ = 0; iPQ < nshell; iPQ++)
                {
                    int dim_iPQ = shell_bf_num[iPQ];
                    if (thread_Mpair_flag[iPQ]) 
                    {
                        int MPQ_blk_ptr = blk_mat_ptr[M * nshell + iPQ];
                        double *K_blk_ptr = K_blk_mat + MPQ_blk_ptr;
                        double *thread_FM_strip_blk_ptr = thread_FM_strip_buf + MPQ_blk_ptr - FM_strip_offset;
                        atomic_add_vector(K_blk_ptr, thread_FM_strip_blk_ptr, dimM * dim_iPQ);
                    }
                    if (thread_Npair_flag[iPQ]) 
                    {
                        int NPQ_blk_ptr = blk_mat_ptr[N * nshell + iPQ];
                        double *K_blk_ptr = K_blk_mat + NPQ_blk_ptr;
                        double *thread_FN_strip_blk_ptr = thread_FN_strip_buf + NPQ_blk_ptr - FN_strip_offset;
                        atomic_add_vector(K_blk_ptr, thread_FN_strip_blk_ptr, dimN * dim_iPQ);
                    }
                }  // End of iPQ loop
            }  // End of "if (build_K)"
        }  // End of MN loop
        
        TinyDFT_JKblkmat_to_JKmat(
            nshell, nbf, shell_bf_num, shell_bf_sind, blk_mat_ptr,
            J_blk_mat, K_blk_mat, J_mat, K_mat, build_J, build_K
        );
        TinyDFT_finalize_JKmat(nbf, J_mat, K_mat, build_J, build_K);
        
        CMS_Simint_free_multi_sp(thread_multi_shellpair);
        free_ThreadKetShellpairLists(thread_ksp_lists);
    }  // End of "#pragma omp parallel"
}

void TinyDFT_calc_HF_energy(
    const int mat_size, const double *D_mat, const double *Hcore_mat, const double *J_mat, 
    const double *K_mat, double *E_one_elec, double *E_two_elec, double *E_HF_exchange
)
{
    double Eoe = 0.0, Ete = 0.0, Exc = 0.0;
    if (K_mat != NULL)
    {
        #pragma omp parallel for simd reduction(+:Eoe, Ete, Exc)
        for (int i = 0; i < mat_size; i++)
        {
            Eoe += D_mat[i] * Hcore_mat[i];
            Ete += D_mat[i] * J_mat[i];
            Exc += D_mat[i] * K_mat[i];
        }
        Eoe *= 2.0;
        Ete *= 2.0;
        Exc *= -1.0;
        *E_one_elec    = Eoe;
        *E_two_elec    = Ete;
        *E_HF_exchange = Exc;
    } else {
        #pragma omp parallel for simd reduction(+:Eoe, Ete)
        for (int i = 0; i < mat_size; i++)
        {
            Eoe += D_mat[i] * Hcore_mat[i];
            Ete += D_mat[i] * J_mat[i];
        }
        Eoe *= 2.0;
        Ete *= 2.0;
        *E_one_elec = Eoe;
        *E_two_elec = Ete;
    }
}
