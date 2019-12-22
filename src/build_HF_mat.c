#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include <mkl.h>

#include "utils.h"
#include "TinySCF_typedef.h"
#include "build_HF_mat.h"
#include "ket_sp_list.h"
#include "acc_JKmat.h"
#include "libCMS.h"

void TinySCF_build_Hcore_S_X_mat(TinySCF_t TinySCF, double *Hcore_mat, double *S_mat, double *X_mat)
{
    assert(TinySCF != NULL);
    
    int nbf            = TinySCF->nbf;
    int nshell         = TinySCF->nshell;
    int mat_size       = TinySCF->mat_size;
    int *shell_bf_sind = TinySCF->shell_bf_sind;
    int *shell_bf_num  = TinySCF->shell_bf_num;
    Simint_t   simint  = TinySCF->simint;
    BasisSet_t basis   = TinySCF->basis;
    
    // Compute core Hamiltonian and overlap matrix
    memset(Hcore_mat, 0, DBL_SIZE * mat_size);
    memset(S_mat,     0, DBL_SIZE * mat_size);
    #pragma omp parallel for schedule(dynamic)
    for (int M = 0; M < nshell; M++)
    {
        int tid = omp_get_thread_num();
        for (int N = 0; N < nshell; N++)
        {
            int nints, offset, nrows, ncols;
            double *integrals, *S_ptr, *Hcore_ptr;
            
            offset    = shell_bf_sind[M] * nbf + shell_bf_sind[N];
            S_ptr     = S_mat  + offset;
            Hcore_ptr = Hcore_mat + offset;
            nrows     = shell_bf_num[M];
            ncols     = shell_bf_num[N];
            
            // Compute the contribution of current shell pair to core Hamiltonian matrix
            CMS_computePairOvl_Simint(basis, simint, tid, M, N, &integrals, &nints);
            if (nints > 0) copy_dbl_mat_blk(S_ptr, nbf, integrals, ncols, nrows, ncols);
            
            // Compute the contribution of current shell pair to overlap matrix
            CMS_computePairCoreH_Simint(basis, simint, tid, M, N, &integrals, &nints);
            if (nints > 0) copy_dbl_mat_blk(Hcore_ptr, nbf, integrals, ncols, nrows, ncols);
        }
    }
    
    // Construct basis transformation matrix
    double *workbuf = (double*) malloc(sizeof(double) * (2 * mat_size + nbf));
    assert(workbuf != NULL);
    double *U_mat  = workbuf; 
    double *U0_mat = U_mat  + mat_size;
    double *eigval = U0_mat + mat_size;
    // [U, D] = eig(S);
    memcpy(U_mat, S_mat, DBL_SIZE * mat_size);
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', nbf, U_mat, nbf, eigval); // U_mat will be overwritten by eigenvectors
    // X = U * D^{-1/2} * U'^T
    memcpy(U0_mat, U_mat, DBL_SIZE * mat_size);
    for (int i = 0; i < nbf; i++) eigval[i] = 1.0 / sqrt(eigval[i]);
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
static void TinySCF_finalize_JKmat(const int nbf, double *J_mat, double *K_mat)
{
    #pragma omp for schedule(dynamic)
    for (int irow = 0; irow < nbf; irow++)
    {
        int idx = irow * nbf + irow;
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

static void TinySCF_JKblkmat_to_JKmat(
    double *J_mat, double *K_mat, double *J_blk_mat, double *K_blk_mat,
    int *blk_mat_ptr, int *shell_bf_num, int *shell_bf_sind, int nshell, int nbf
)
{
    #pragma omp for
    for (int i = 0; i < nshell; i++)
    {
        for (int j = 0; j < nshell; j++)
        {
            int Jblk_offset = blk_mat_ptr[i * nshell + j];
            int J_offset    = shell_bf_sind[i] * nbf + shell_bf_sind[j];
            copy_dbl_mat_blk(
                J_mat + J_offset, nbf, J_blk_mat + Jblk_offset, shell_bf_num[j],
                shell_bf_num[i], shell_bf_num[j]
            );
        }
    }
    
    #pragma omp for
    for (int i = 0; i < nshell; i++)
    {
        for (int j = 0; j < nshell; j++)
        {
            int Kblk_offset = blk_mat_ptr[i * nshell + j];
            int K_offset    = shell_bf_sind[i] * nbf + shell_bf_sind[j];
            copy_dbl_mat_blk(
                K_mat + K_offset, nbf, K_blk_mat + Kblk_offset, shell_bf_num[j],
                shell_bf_num[i], shell_bf_num[j]
            );
        }
    }
}

static void TinySCF_Dmat_to_Dblkmat(
    const double *D_mat, double *D_blk_mat, int *blk_mat_ptr, 
    int *shell_bf_num, int *shell_bf_sind, int nshell, int nbf
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
                D_blk_mat + Dblk_offset, shell_bf_num[j], D_mat + D_offset, nbf, 
                shell_bf_num[i], shell_bf_num[j]
            );
        }
    }
}

void TinySCF_build_JKmat(TinySCF_t TinySCF, const double *D_mat, double *J_mat, double *K_mat)
{
    int    nbf            = TinySCF->nbf;
    int    nshell         = TinySCF->nshell;
    int    num_valid_sp   = TinySCF->num_valid_sp;
    int    max_dim        = TinySCF->max_dim;
    int    mat_size       = TinySCF->mat_size;
    int    max_JKacc_buf  = TinySCF->max_JKacc_buf;
    int    *shell_bf_num  = TinySCF->shell_bf_num;
    int    *shell_bf_sind = TinySCF->shell_bf_sind;
    int    *valid_sp_lid  = TinySCF->valid_sp_lid;
    int    *valid_sp_rid  = TinySCF->valid_sp_rid;
    int    *blk_mat_ptr   = TinySCF->blk_mat_ptr;
    int    *Mpair_flag    = TinySCF->Mpair_flag;
    int    *Npair_flag    = TinySCF->Npair_flag;
    double scrtol2        = TinySCF->shell_scrtol2;
    double *sp_scrval     = TinySCF->sp_scrval;
    double *J_blk_mat     = TinySCF->J_blk_mat;
    double *K_blk_mat     = TinySCF->K_blk_mat;
    double *D_blk_mat     = TinySCF->D_blk_mat;
    double *FM_strip_buf  = TinySCF->FM_strip_buf;
    double *FN_strip_buf  = TinySCF->FN_strip_buf;
    Simint_t simint       = TinySCF->simint;
    
    memset(J_blk_mat, 0, DBL_SIZE * mat_size);
    memset(K_blk_mat, 0, DBL_SIZE * mat_size);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        TinySCF_Dmat_to_Dblkmat(
            D_mat, D_blk_mat, blk_mat_ptr,
            shell_bf_num, shell_bf_sind, nshell, nbf
        );
        
        // Create ERI batching auxiliary data structures
        // Ket-side shell pair lists that needs to be computed
        ThreadKetShellpairLists_t thread_ksp_lists;
        create_ThreadKetShellpairLists(&thread_ksp_lists);
        // Simint multi_shellpair buffer for batched ERI computation
        void *thread_multi_shellpair;
        CMS_Simint_createThreadMultishellpair(&thread_multi_shellpair);
        
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
            
            double *J_MN_buf = TinySCF->JKacc_buf + tid * max_JKacc_buf;
            double *J_MN = J_blk_mat + blk_mat_ptr[M * nshell + N];
            int dimM = shell_bf_num[M], dimN = shell_bf_num[N];
            memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
            
            memset(thread_FM_strip_buf, 0, sizeof(double) * nbf * max_dim);
            memset(thread_FN_strip_buf, 0, sizeof(double) * nbf * max_dim);
            memset(thread_Mpair_flag,  0, sizeof(int)    * nshell);
            memset(thread_Npair_flag,  0, sizeof(int)    * nshell);
            
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
                int ket_id = CMS_Simint_getShellpairAMIndex(simint, P, Q);
                KetShellpairList_t target_shellpair_list = &thread_ksp_lists->ket_shellpair_lists[ket_id];
                add_shellpair_to_KetShellPairList(target_shellpair_list, P, Q);
                
                // If the ket-side shell pair list we just used is full, handle it
                if (target_shellpair_list->npairs == MAX_LIST_SIZE)
                {
                    double *thread_batch_eris;
                    int thread_nints;
                    
                    // Compute batched ERIs
                    CMS_computeShellQuartetBatch_Simint(
                        simint, tid, M, N,
                        target_shellpair_list->P_list,
                        target_shellpair_list->Q_list,
                        target_shellpair_list->npairs,
                        &thread_batch_eris, &thread_nints, 
                        &thread_multi_shellpair
                    );
                    
                    // Accumulate ERI results to global matrices
                    if (thread_nints > 0)
                    {
                        double st = get_wtime_sec();
                        acc_JKmat_with_ket_sp_list(
                            TinySCF, tid, M, N, 
                            target_shellpair_list->P_list,
                            target_shellpair_list->Q_list,
                            target_shellpair_list->npairs,
                            thread_batch_eris, thread_nints,
                            thread_FM_strip_buf, thread_FN_strip_buf,
                            thread_Mpair_flag, thread_Npair_flag
                        );
                        double et = get_wtime_sec();
                        if (tid == 0) CMS_Simint_addupdateFtimer(simint, et - st);
                    }
                    
                    // Reset the computed ket-side shell pair list
                    target_shellpair_list->npairs = 0;
                }  // End of "if (target_shellpair_list->npairs == MAX_LIST_SIZE)"
            }  // End of PQ loop
            
            // Handles all non-empty ket-side shell pair lists
            for (int ket_id = 0; ket_id < MAX_AM_PAIRS; ket_id++)
            {
                KetShellpairList_t target_shellpair_list = &thread_ksp_lists->ket_shellpair_lists[ket_id];
                
                if (target_shellpair_list->npairs > 0)
                {
                    double *thread_batch_eris;
                    int thread_nints;
                    
                    // Compute batched ERIs
                    CMS_computeShellQuartetBatch_Simint(
                        simint, tid, M, N,
                        target_shellpair_list->P_list,
                        target_shellpair_list->Q_list,
                        target_shellpair_list->npairs,
                        &thread_batch_eris, &thread_nints, 
                        &thread_multi_shellpair
                    );
                    
                    // Accumulate ERI results to global matrices
                    if (thread_nints > 0)
                    {
                        double st = get_wtime_sec();
                        acc_JKmat_with_ket_sp_list(
                            TinySCF, tid, M, N, 
                            target_shellpair_list->P_list,
                            target_shellpair_list->Q_list,
                            target_shellpair_list->npairs,
                            thread_batch_eris, thread_nints,
                            thread_FM_strip_buf, thread_FN_strip_buf,
                            thread_Mpair_flag, thread_Npair_flag
                        );
                        double et = get_wtime_sec();
                        if (tid == 0) CMS_Simint_addupdateFtimer(simint, et - st);
                    }
                    
                    // Reset the computed ket-side shell pair list
                    target_shellpair_list->npairs = 0;
                }  // End of "if (target_shellpair_list->npairs > 0)"
            }  // End of ket_id loop
            
            // Accumulate thread-local J and K results to global J and K mat
            atomic_add_vector(J_MN, J_MN_buf, dimM * dimN);
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
            }
        }  // End of MN loop
        
        TinySCF_JKblkmat_to_JKmat(
            J_mat, K_mat, J_blk_mat, K_blk_mat, 
            blk_mat_ptr, shell_bf_num, shell_bf_sind, nshell, nbf
        );
        TinySCF_finalize_JKmat(nbf, J_mat, K_mat);
        
        CMS_Simint_freeThreadMultishellpair(&thread_multi_shellpair);
        free_ThreadKetShellpairLists(thread_ksp_lists);
    }  // End of "#pragma omp parallel"
}

void TinySCF_calc_HF_energy(
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
