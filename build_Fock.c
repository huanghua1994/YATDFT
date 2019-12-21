#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "utils.h"
#include "TinySCF.h"
#include "build_Fock.h"
#include "shell_quartet_list.h"
#include "Accum_Fock.h"
#include "libCMS.h"

#define ACCUM_FOCK_PARAM    TinySCF, tid, M, N, P_list[ipair], Q_list[ipair], \
                            thread_eris + ipair * thread_nints, load_P, write_P, \
                            thread_F_M_band_blocks, thread_M_bank_offset, \
                            thread_F_N_band_blocks, thread_N_bank_offset

void Accum_Fock_with_KetshellpairList(
    TinySCF_t TinySCF, int tid, int M, int N, 
    int *P_list, int *Q_list, int npairs, 
    double *thread_eris, int thread_nints,
    double *thread_F_M_band_blocks, double *thread_F_N_band_blocks,
    int *thread_visited_Mpairs, int *thread_visited_Npairs
)
{
    int load_P, write_P, prev_P = -1;
    int dimM = TinySCF->shell_bf_num[M];
    int dimN = TinySCF->shell_bf_num[N];
    int thread_M_bank_offset = TinySCF->mat_block_ptr[M * TinySCF->nshells];
    int thread_N_bank_offset = TinySCF->mat_block_ptr[N * TinySCF->nshells];
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
        
        thread_visited_Mpairs[curr_P] = 1;
        thread_visited_Mpairs[curr_Q] = 1;
        thread_visited_Npairs[curr_P] = 1;
        thread_visited_Npairs[curr_Q] = 1;
        
        int dimP = TinySCF->shell_bf_num[curr_P];
        int dimQ = TinySCF->shell_bf_num[curr_Q];
        int is_1111 = dimM * dimN * dimP * dimQ;
        
        if (is_1111 == 1)    Accum_Fock_1111  (ACCUM_FOCK_PARAM);
        else if (dimQ == 1)  Accum_Fock_dimQ1 (ACCUM_FOCK_PARAM);
        else if (dimQ == 3)  Accum_Fock_dimQ3 (ACCUM_FOCK_PARAM);
        else if (dimQ == 6)  Accum_Fock_dimQ6 (ACCUM_FOCK_PARAM);
        else if (dimQ == 10) Accum_Fock_dimQ10(ACCUM_FOCK_PARAM);
        else if (dimQ == 15) Accum_Fock_dimQ15(ACCUM_FOCK_PARAM);
        else Accum_Fock(ACCUM_FOCK_PARAM);
    }
}

// F = H_core + (J + J^T) / 2 + (K + K^T) / 2;
static void TinySCF_HJKmat_to_Fmat(double *Hcore_mat, double *J_mat, double *K_mat, double *F_mat, int nbf)
{
    #pragma omp for
    for (int irow = 0; irow < nbf; irow++)
    {
        int idx = irow * nbf + irow;
        F_mat[idx] = Hcore_mat[idx] + J_mat[idx] + K_mat[idx];
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
            F_mat[idx1] = Hcore_mat[idx1] + Jval + Kval;
            F_mat[idx2] = Hcore_mat[idx2] + Jval + Kval;
        }
    }
}

static void TinySCF_JKmatblock_to_JKmat(
    double *J_mat, double *K_mat, double *J_mat_block, double *K_mat_block,
    int *mat_block_ptr, int *shell_bf_num, int *shell_bf_sind, int nshells, int nbf
)
{    
    #pragma omp for
    for (int i = 0; i < nshells; i++)
    {
        for (int j = 0; j < nshells; j++)
        {
            int mat_block_pos  = mat_block_ptr[i * nshells + j];
            int global_mat_pos = shell_bf_sind[i] * nbf + shell_bf_sind[j];
            #ifdef BUILD_J_MAT_STD
            copy_matrix_block(
                J_mat + global_mat_pos, nbf, 
                J_mat_block + mat_block_pos, shell_bf_num[j],
                shell_bf_num[i], shell_bf_num[j]
            );
            #endif
            #ifdef BUILD_K_MAT_HF
            copy_matrix_block(
                K_mat + global_mat_pos, nbf, 
                K_mat_block + mat_block_pos, shell_bf_num[j],
                shell_bf_num[i], shell_bf_num[j]
            );
            #endif
        }
    }
}

static void TinySCF_Dmat_to_Dmatblock(
    double *D_mat, double *D_mat_block,    int *mat_block_ptr, 
    int *shell_bf_num, int *shell_bf_sind, int nshells, int nbf
)
{
    #pragma omp for
    for (int i = 0; i < nshells; i++)
    {
        for (int j = 0; j < nshells; j++)
        {
            int mat_block_pos  = mat_block_ptr[i * nshells + j];
            int global_mat_pos = shell_bf_sind[i] * nbf + shell_bf_sind[j];
            copy_matrix_block(
                D_mat_block + mat_block_pos, shell_bf_num[j],
                D_mat + global_mat_pos, nbf, 
                shell_bf_num[i], shell_bf_num[j]
            );
        }
    }
}

void TinySCF_build_FockMat(TinySCF_t TinySCF)
{
    // Copy some parameters out, I don't want to see so many "TinySCF->"
    int nshells         = TinySCF->nshells;
    int num_uniq_sp     = TinySCF->num_uniq_sp;
    int max_dim         = TinySCF->max_dim;
    double scrtol2      = TinySCF->shell_scrtol2;
    double *sp_scrval   = TinySCF->sp_scrval;
    int *shell_bf_num   = TinySCF->shell_bf_num;
    int *shell_bf_sind  = TinySCF->shell_bf_sind;
    int *uniq_sp_lid    = TinySCF->uniq_sp_lid;
    int *uniq_sp_rid    = TinySCF->uniq_sp_rid;
    int num_bas_func    = TinySCF->nbasfuncs;
    Simint_t simint     = TinySCF->simint;
    double *J_mat       = TinySCF->J_mat;
    double *K_mat       = TinySCF->K_mat;
    double *F_mat       = TinySCF->F_mat;
    double *D_mat       = TinySCF->D_mat;
    double *Hcore_mat   = TinySCF->Hcore_mat;
    int *mat_block_ptr  = TinySCF->mat_block_ptr;
    double *J_mat_block = TinySCF->J_mat_block;
    double *K_mat_block = TinySCF->K_mat_block;
    double *D_mat_block = TinySCF->D_mat_block;
    
    memset(J_mat_block, 0, DBL_SIZE * TinySCF->mat_size);
    memset(K_mat_block, 0, DBL_SIZE * TinySCF->mat_size);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        TinySCF_Dmat_to_Dmatblock(
            D_mat, D_mat_block, mat_block_ptr,
            shell_bf_num, shell_bf_sind, nshells, num_bas_func
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
        int    *thread_visited_Mpairs  = TinySCF->visited_Mpairs  + tid * nshells;
        int    *thread_visited_Npairs  = TinySCF->visited_Npairs  + tid * nshells;
        
        #pragma omp for schedule(dynamic)
        for (int MN = 0; MN < num_uniq_sp; MN++)
        {
            int M = uniq_sp_lid[MN];
            int N = uniq_sp_rid[MN];
            double scrval1 = sp_scrval[M * nshells + N];
            
            double *J_MN_buf = TinySCF->Accum_Fock_buf + tid * TinySCF->max_buf_size;
            double *J_MN = J_mat_block + mat_block_ptr[M * nshells + N];
            int dimM = shell_bf_num[M], dimN = shell_bf_num[N];
            memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
            
            memset(thread_F_M_band_blocks, 0, sizeof(double) * num_bas_func * max_dim);
            memset(thread_F_N_band_blocks, 0, sizeof(double) * num_bas_func * max_dim);
            memset(thread_visited_Mpairs,  0, sizeof(int)    * nshells);
            memset(thread_visited_Npairs,  0, sizeof(int)    * nshells);
            
            for (int PQ = 0; PQ < num_uniq_sp; PQ++)
            {
                int P = uniq_sp_lid[PQ];
                int Q = uniq_sp_rid[PQ];
                double scrval2 = sp_scrval[P * nshells + Q];
                
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
            int thread_M_bank_offset = mat_block_ptr[M * nshells];
            int thread_N_bank_offset = mat_block_ptr[N * nshells];
            for (int iPQ = 0; iPQ < nshells; iPQ++)
            {
                int dim_iPQ = shell_bf_num[iPQ];
                if (thread_visited_Mpairs[iPQ]) 
                {
                    int MPQ_block_ptr = mat_block_ptr[M * nshells + iPQ];
                    double *global_K_block_ptr = K_mat_block + MPQ_block_ptr;
                    double *thread_F_M_band_block_ptr = thread_F_M_band_blocks + MPQ_block_ptr - thread_M_bank_offset;
                    atomic_add_vector(global_K_block_ptr, thread_F_M_band_block_ptr, dimM * dim_iPQ);
                }
                if (thread_visited_Npairs[iPQ]) 
                {
                    int NPQ_block_ptr = mat_block_ptr[N * nshells + iPQ];
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
        
        TinySCF_JKmatblock_to_JKmat(
            J_mat, K_mat, J_mat_block, K_mat_block, 
            mat_block_ptr, shell_bf_num, shell_bf_sind, nshells, num_bas_func
        );
        TinySCF_HJKmat_to_Fmat(Hcore_mat, J_mat, K_mat, F_mat, num_bas_func);
    }
}
