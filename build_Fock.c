#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include <mkl.h>

#include "utils.h"
#include "TinySCF.h"
#include "build_Fock.h"
#include "shell_quartet_list.h"
#include "Accum_Fock.h"
#include "libCMS.h"

void TinySCF_build_Hcore_S_X(TinySCF_t TinySCF, double *Hc, double *S, double *X)
{
    assert(TinySCF != NULL);
    
    int nbf            = TinySCF->nbasfuncs;
    int nshell         = TinySCF->nshells;
    int mat_size       = TinySCF->mat_size;
    int *shell_bf_sind = TinySCF->shell_bf_sind;
    int *shell_bf_num  = TinySCF->shell_bf_num;
    Simint_t   simint  = TinySCF->simint;
    BasisSet_t basis   = TinySCF->basis;
    
    // Compute core Hamiltonian and overlap matrix
    memset(Hc, 0, DBL_SIZE * mat_size);
    memset(S,  0, DBL_SIZE * mat_size);
    #pragma omp parallel for schedule(dynamic)
    for (int M = 0; M < nshell; M++)
    {
        int tid = omp_get_thread_num();
        for (int N = 0; N < nshell; N++)
        {
            int nints, offset, nrows, ncols;
            double *integrals, *S_ptr, *Hc_ptr;
            
            offset = shell_bf_sind[M] * nbf + shell_bf_sind[N];
            S_ptr  = S  + offset;
            Hc_ptr = Hc + offset;
            nrows  = shell_bf_num[M];
            ncols  = shell_bf_num[N];
            
            // Compute the contribution of current shell pair to core Hamiltonian matrix
            CMS_computePairOvl_Simint(basis, simint, tid, M, N, &integrals, &nints);
            if (nints > 0) copy_matrix_block(S_ptr, nbf, integrals, ncols, nrows, ncols);
            
            // Compute the contribution of current shell pair to overlap matrix
            CMS_computePairCoreH_Simint(basis, simint, tid, M, N, &integrals, &nints);
            if (nints > 0) copy_matrix_block(Hc_ptr, nbf, integrals, ncols, nrows, ncols);
        }
    }
    
    // Construct basis transformation matrix
    double *workbuf = (double*) malloc(sizeof(double) * (2 * mat_size + nbf));
    assert(workbuf != NULL);
    double *U  = workbuf; 
    double *U0 = U  + mat_size;
    double *ev = U0 + mat_size;
    // [U, D] = eig(S);
    memcpy(U, S, DBL_SIZE * mat_size);
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', nbf, U, nbf, ev); // U will be overwritten by eigenvectors
    // X = U * D^{-1/2} * U'^T
    memcpy(U0, U, DBL_SIZE * mat_size);
    for (int i = 0; i < nbf; i++) ev[i] = 1.0 / sqrt(ev[i]);
    for (int i = 0; i < nbf; i++)
    {
        #pragma omp simd
        for (int j = 0; j < nbf; j++)
            U0[i * nbf + j] *= ev[j];
    }
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, nbf, nbf, nbf, 
        1.0, U0, nbf, U, nbf, 0.0, X, nbf
    );
    free(workbuf);
}

#define ACCUM_FOCK_PARAM    TinySCF, tid, M, N, P_list[ipair], Q_list[ipair], \
                            ERIs + ipair * nints, load_P, write_P, \
                            FM_strip_blocks, FM_strip_offset, \
                            FN_strip_blocks, FN_strip_offset

void Accum_Fock_with_KetshellpairList(
    TinySCF_t TinySCF, int tid, int M, int N, 
    int *P_list, int *Q_list, int npairs, double *ERIs, int nints,
    double *FM_strip_blocks, double *FN_strip_blocks,
    int *visited_Mpairs, int *visited_Npairs
)
{
    int nshell = TinySCF->nshells;
    int *mat_blk_ptr = TinySCF->mat_block_ptr;
    int load_P, write_P, prev_P = -1;
    int dimM = TinySCF->shell_bf_num[M];
    int dimN = TinySCF->shell_bf_num[N];
    int FM_strip_offset = mat_blk_ptr[M * nshell];
    int FN_strip_offset = mat_blk_ptr[N * nshell];
    for (int ipair = 0; ipair < npairs; ipair++)
    {
        int curr_P = P_list[ipair];
        int curr_Q = Q_list[ipair];
        
        load_P  = (prev_P == curr_P) ? 0 : 1;
        
        write_P = 0;
        if (ipair == npairs - 1)
        {
            write_P = 1;
        } else {
            if (curr_P != P_list[ipair + 1]) write_P = 1;
        }
        prev_P = curr_P;
        
        visited_Mpairs[curr_P] = 1;
        visited_Mpairs[curr_Q] = 1;
        visited_Npairs[curr_P] = 1;
        visited_Npairs[curr_Q] = 1;
        
        int dimP = TinySCF->shell_bf_num[curr_P];
        int dimQ = TinySCF->shell_bf_num[curr_Q];
        int is_1111 = dimM * dimN * dimP * dimQ;
        
        Accum_Fock(ACCUM_FOCK_PARAM);
    }
}

// Get the final J and K matrices: J = (J + J^T) / 2, K = (K + K^T) / 2
static void TinySCF_finalize_JK(const int nbf, double *J, double *K)
{
    #pragma omp for schedule(dynamic)
    for (int irow = 0; irow < nbf; irow++)
    {
        int idx = irow * nbf + irow;
        for (int icol = irow + 1; icol < nbf; icol++)
        {
            int idx1 = irow * nbf + icol;
            int idx2 = icol * nbf + irow;
            double Jval = (J[idx1] + J[idx2]) * 0.5;
            double Kval = (K[idx1] + K[idx2]) * 0.5;
            J[idx1] = Jval;
            J[idx2] = Jval;
            K[idx1] = Kval;
            K[idx2] = Kval;
        }
    }
}

static void TinySCF_JKblk_to_JK(
    double *J, double *K, double *Jblk, double *Kblk,
    int *mat_block_ptr, int *shell_bf_num, int *shell_bf_sind, int nshell, int nbf
)
{
    #ifdef BUILD_J_MAT_STD
    #pragma omp for
    for (int i = 0; i < nshell; i++)
    {
        for (int j = 0; j < nshell; j++)
        {
            int Jblk_offset = mat_block_ptr[i * nshell + j];
            int J_offset    = shell_bf_sind[i] * nbf + shell_bf_sind[j];
            copy_matrix_block(
                J + J_offset, nbf, Jblk + Jblk_offset, shell_bf_num[j],
                shell_bf_num[i], shell_bf_num[j]
            );
        }
    }
    #endif
    
    #ifdef BUILD_K_MAT_HF
    #pragma omp for
    for (int i = 0; i < nshell; i++)
    {
        for (int j = 0; j < nshell; j++)
        {
            int Kblk_offset = mat_block_ptr[i * nshell + j];
            int K_offset    = shell_bf_sind[i] * nbf + shell_bf_sind[j];
            copy_matrix_block(
                K + K_offset, nbf, Kblk + Kblk_offset, shell_bf_num[j],
                shell_bf_num[i], shell_bf_num[j]
            );
        }
    }
    #endif
}

static void TinySCF_D_to_Dblk(
    double *D, double *Dblk, int *mat_block_ptr, 
    int *shell_bf_num, int *shell_bf_sind, int nshell, int nbf
)
{
    #pragma omp for
    for (int i = 0; i < nshell; i++)
    {
        for (int j = 0; j < nshell; j++)
        {
            int Dblk_offset = mat_block_ptr[i * nshell + j];
            int D_offset    = shell_bf_sind[i] * nbf + shell_bf_sind[j];
            copy_matrix_block(
                Dblk + Dblk_offset, shell_bf_num[j], D + D_offset, nbf, 
                shell_bf_num[i], shell_bf_num[j]
            );
        }
    }
}

void TinySCF_build_FockMat(TinySCF_t TinySCF)
{
    int nshell          = TinySCF->nshells;
    int num_uniq_sp     = TinySCF->num_uniq_sp;
    int max_dim         = TinySCF->max_dim;
    int num_bas_func    = TinySCF->nbasfuncs;
    int mat_size        = TinySCF->mat_size;
    int *shell_bf_num   = TinySCF->shell_bf_num;
    int *shell_bf_sind  = TinySCF->shell_bf_sind;
    int *uniq_sp_lid    = TinySCF->uniq_sp_lid;
    int *uniq_sp_rid    = TinySCF->uniq_sp_rid;
    int *mat_block_ptr  = TinySCF->mat_block_ptr;
    double scrtol2      = TinySCF->shell_scrtol2;
    double *sp_scrval   = TinySCF->sp_scrval;
    double *J_mat       = TinySCF->J_mat;
    double *K_mat       = TinySCF->K_mat;
    double *F_mat       = TinySCF->F_mat;
    double *D_mat       = TinySCF->D_mat;
    double *Hcore_mat   = TinySCF->Hcore_mat;
    double *J_mat_block = TinySCF->J_mat_block;
    double *K_mat_block = TinySCF->K_mat_block;
    double *D_mat_block = TinySCF->D_mat_block;
    Simint_t simint     = TinySCF->simint;
    
    memset(J_mat_block, 0, DBL_SIZE * TinySCF->mat_size);
    memset(K_mat_block, 0, DBL_SIZE * TinySCF->mat_size);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        TinySCF_D_to_Dblk(
            D_mat, D_mat_block, mat_block_ptr,
            shell_bf_num, shell_bf_sind, nshell, num_bas_func
        );
        
        // Create ERI batching auxiliary data structures
        // Ket-side shell pair lists that needs to be computed
        ThreadKetShellpairLists_t thread_ksp_lists;
        create_ThreadKetShellpairLists(&thread_ksp_lists);
        // Simint multi_shellpair buffer for batched ERI computation
        void *thread_multi_shellpair;
        CMS_Simint_createThreadMultishellpair(&thread_multi_shellpair);
        
        double *thread_F_M_band_blocks = TinySCF->F_M_band_blocks + tid * num_bas_func * max_dim;
        double *thread_F_N_band_blocks = TinySCF->F_N_band_blocks + tid * num_bas_func * max_dim;
        int    *thread_visited_Mpairs  = TinySCF->visited_Mpairs  + tid * nshell;
        int    *thread_visited_Npairs  = TinySCF->visited_Npairs  + tid * nshell;
        
        #pragma omp for schedule(dynamic)
        for (int MN = 0; MN < num_uniq_sp; MN++)
        {
            int M = uniq_sp_lid[MN];
            int N = uniq_sp_rid[MN];
            double scrval1 = sp_scrval[M * nshell + N];
            
            double *J_MN_buf = TinySCF->Accum_Fock_buf + tid * TinySCF->max_buf_size;
            double *J_MN = J_mat_block + mat_block_ptr[M * nshell + N];
            int dimM = shell_bf_num[M], dimN = shell_bf_num[N];
            memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
            
            memset(thread_F_M_band_blocks, 0, sizeof(double) * num_bas_func * max_dim);
            memset(thread_F_N_band_blocks, 0, sizeof(double) * num_bas_func * max_dim);
            memset(thread_visited_Mpairs,  0, sizeof(int)    * nshell);
            memset(thread_visited_Npairs,  0, sizeof(int)    * nshell);
            
            for (int PQ = 0; PQ < num_uniq_sp; PQ++)
            {
                int P = uniq_sp_lid[PQ];
                int Q = uniq_sp_rid[PQ];
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
                        Accum_Fock_with_KetshellpairList(
                            TinySCF, tid, M, N, 
                            target_shellpair_list->P_list,
                            target_shellpair_list->Q_list,
                            target_shellpair_list->npairs,
                            thread_batch_eris, thread_nints,
                            thread_F_M_band_blocks, thread_F_N_band_blocks,
                            thread_visited_Mpairs, thread_visited_Npairs
                        );
                        double et = get_wtime_sec();
                        if (tid == 0) CMS_Simint_addupdateFtimer(simint, et - st);
                    }
                    
                    // Reset the computed ket-side shell pair list
                    target_shellpair_list->npairs = 0;
                }  // if (target_shellpair_list->npairs == MAX_LIST_SIZE)
            }  // for (int PQ = 0; PQ < num_uniq_sp; PQ++)
            
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
                        Accum_Fock_with_KetshellpairList(
                            TinySCF, tid, M, N, 
                            target_shellpair_list->P_list,
                            target_shellpair_list->Q_list,
                            target_shellpair_list->npairs,
                            thread_batch_eris, thread_nints,
                            thread_F_M_band_blocks, thread_F_N_band_blocks,
                            thread_visited_Mpairs, thread_visited_Npairs
                        );
                        double et = get_wtime_sec();
                        if (tid == 0) CMS_Simint_addupdateFtimer(simint, et - st);
                    }
                    
                    // Reset the computed ket-side shell pair list
                    target_shellpair_list->npairs = 0;
                }  // if (target_shellpair_list->npairs > 0)
            }  // for (int ket_id = 0; ket_id < MAX_AM_PAIRS; ket_id++)
            
            #ifdef BUILD_J_MAT_STD
            atomic_add_vector(J_MN, J_MN_buf, dimM * dimN);
            #endif
            #ifdef BUILD_K_MAT_HF
            int thread_M_bank_offset = mat_block_ptr[M * nshell];
            int thread_N_bank_offset = mat_block_ptr[N * nshell];
            for (int iPQ = 0; iPQ < nshell; iPQ++)
            {
                int dim_iPQ = shell_bf_num[iPQ];
                if (thread_visited_Mpairs[iPQ]) 
                {
                    int MPQ_block_ptr = mat_block_ptr[M * nshell + iPQ];
                    double *global_K_block_ptr = K_mat_block + MPQ_block_ptr;
                    double *thread_F_M_band_block_ptr = thread_F_M_band_blocks + MPQ_block_ptr - thread_M_bank_offset;
                    atomic_add_vector(global_K_block_ptr, thread_F_M_band_block_ptr, dimM * dim_iPQ);
                }
                if (thread_visited_Npairs[iPQ]) 
                {
                    int NPQ_block_ptr = mat_block_ptr[N * nshell + iPQ];
                    double *global_K_block_ptr = K_mat_block + NPQ_block_ptr;
                    double *thread_F_N_band_block_ptr = thread_F_N_band_blocks + NPQ_block_ptr - thread_N_bank_offset;
                    atomic_add_vector(global_K_block_ptr, thread_F_N_band_block_ptr, dimN * dim_iPQ);
                }
            }
            #endif
        }  // for (int MN = 0; MN < num_uniq_sp; MN++)
        
        // Free batch ERI auxiliary data structures
        CMS_Simint_freeThreadMultishellpair(&thread_multi_shellpair);
        free_ThreadKetShellpairLists(thread_ksp_lists);
        
        TinySCF_JKblk_to_JK(
            J_mat, K_mat, J_mat_block, K_mat_block, 
            mat_block_ptr, shell_bf_num, shell_bf_sind, nshell, num_bas_func
        );
        
        TinySCF_finalize_JK(num_bas_func, J_mat, K_mat);
        
        #pragma omp for simd
        for (int i = 0; i < mat_size; i++)
            F_mat[i] = Hcore_mat[i] + 2 * J_mat[i] - K_mat[i];
    }
}
