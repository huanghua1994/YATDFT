#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "TinyDFT_typedef.h"
#include "acc_JKmat.h"

static inline void unique_integral_coef(int M, int N, int P, int Q, double *coef)
{
    int flag1 = (M == N) ? 0 : 1;
    int flag2 = (P == Q) ? 0 : 1;
    int flag3 = ((M == P) && (N == Q)) ? 0 : 1;
    int flag4 = ((flag1 == 1) && (flag2 == 1)) ? 1 : 0;
    int flag5 = ((flag1 == 1) && (flag3 == 1)) ? 1 : 0;
    int flag6 = ((flag2 == 1) && (flag3 == 1)) ? 1 : 0;
    int flag7 = ((flag4 == 1) && (flag3 == 1)) ? 1 : 0;
    coef[0] = 1.0   + flag1 + flag2 + flag4;  // for J_MN
    coef[1] = flag3 + flag5 + flag6 + flag7;  // for J_PQ
    coef[2] = 1.0   + flag3;  // for K_MP
    coef[3] = flag1 + flag5;  // for K_NP
    coef[4] = flag2 + flag6;  // for K_MQ
    coef[5] = flag4 + flag7;  // for K_NQ
}

static inline void update_global_JKblk(
    int dimM, int dimN, int dimP, int dimQ, int write_P,
    double *K_MP, double *K_MP_buf, double *K_NP, double *K_NP_buf, 
    double *J_PQ, double *J_PQ_buf, double *K_MQ, double *K_MQ_buf, 
    double *K_NQ, double *K_NQ_buf, int build_J,  int build_K
)
{
    if (build_J) atomic_add_vector(J_PQ, J_PQ_buf, dimP * dimQ);
    if (build_K)
    {
        if (write_P)
        {
            direct_add_vector(K_MP, K_MP_buf, dimM * dimP);
            direct_add_vector(K_NP, K_NP_buf, dimN * dimP);
        }
        direct_add_vector(K_MQ, K_MQ_buf, dimM * dimQ);
        direct_add_vector(K_NQ, K_NQ_buf, dimN * dimQ);
    }
}

void acc_JKmat(ACC_JKMAT_IN_PARAM)
{
    int nshell        = TinyDFT->nshell;
    int max_JKacc_buf = TinyDFT->max_JKacc_buf;
    int *shell_bf_num = TinyDFT->shell_bf_num;
    int *blk_mat_ptr  = TinyDFT->blk_mat_ptr;
    double *J_blk_mat = TinyDFT->J_blk_mat;
    double *D_blk_mat = TinyDFT->D_blk_mat;
    double *JKacc_buf = TinyDFT->JKacc_buf;
    
    // Set matrix size info
    int dimM = shell_bf_num[M];
    int dimN = shell_bf_num[N];
    int dimP = shell_bf_num[P];
    int dimQ = shell_bf_num[Q];
    
    // Set global matrix pointers
    double *J_PQ = J_blk_mat + blk_mat_ptr[P * nshell + Q];
    double *K_MP = FM_strip_buf + (blk_mat_ptr[M * nshell + P] - FM_strip_offset); 
    double *K_NP = FN_strip_buf + (blk_mat_ptr[N * nshell + P] - FN_strip_offset);
    double *K_MQ = FM_strip_buf + (blk_mat_ptr[M * nshell + Q] - FM_strip_offset);
    double *K_NQ = FN_strip_buf + (blk_mat_ptr[N * nshell + Q] - FN_strip_offset);
    
    double *D_MN = D_blk_mat + blk_mat_ptr[M * nshell + N];
    double *D_PQ = D_blk_mat + blk_mat_ptr[P * nshell + Q];
    double *D_MP = D_blk_mat + blk_mat_ptr[M * nshell + P];
    double *D_NP = D_blk_mat + blk_mat_ptr[N * nshell + P];
    double *D_MQ = D_blk_mat + blk_mat_ptr[M * nshell + Q];
    double *D_NQ = D_blk_mat + blk_mat_ptr[N * nshell + Q];
    
    // Set buffer pointer
    double *thread_buf = JKacc_buf + tid * max_JKacc_buf;
    int required_buf_size = (dimP + dimN + dimM) * dimQ + (dimP + dimN + dimM) * dimQ;
    assert(required_buf_size <= max_JKacc_buf);
    double *write_buf = thread_buf;
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * dimQ;
    double *K_NQ_buf = write_buf;  write_buf += dimN * dimQ;
    double *K_MQ_buf = write_buf;  write_buf += dimM * dimQ;

    // Reset result buffer
    //if (load_MN) memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
    if (load_P) memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
    memset(J_PQ_buf, 0, sizeof(double) * dimQ * (dimM + dimN + dimP));
    
    // Get uniqueness ERI symmetric 
    double coef[6];
    unique_integral_coef(M, N, P, Q, coef);
    
    for (int iM = 0; iM < dimM; iM++) 
    {
        int iM_dimN = iM * dimN;
        int iM_dimP = iM * dimP;
        int iM_dimQ = iM * dimQ;
        for (int iN = 0; iN < dimN; iN++) 
        {
            int iN_dimP = iN * dimP;
            int iN_dimQ = iN * dimQ;
            double coef1_D_MN = coef[1] * D_MN[iM_dimN + iN];
            double j_MN = 0.0;
            for (int iP = 0; iP < dimP; iP++) 
            {
                int iP_dimQ = iP * dimQ;
                int Ibase = dimQ * (iP + dimP * (iN + dimN * iM));
                double ncoef4_D_NP = coef[4] * D_NP[iN_dimP + iP];
                double ncoef5_D_MP = coef[5] * D_MP[iM_dimP + iP];
                double k_MP = 0.0, k_NP = 0.0;
                // dimQ is small, vectorizing short loops may hurt performance 
                // since it needs horizon reduction after the loop
                for (int iQ = 0; iQ < dimQ; iQ++) 
                {
                    double eri = ERI[Ibase + iQ];
                    
                    j_MN += D_PQ[iP_dimQ + iQ] * eri;
                    k_MP += D_NQ[iN_dimQ + iQ] * eri;
                    k_NP += D_MQ[iM_dimQ + iQ] * eri;
                    
                    J_PQ_buf[iP_dimQ + iQ] +=  coef1_D_MN * eri;
                    K_MQ_buf[iM_dimQ + iQ] += ncoef4_D_NP * eri;
                    K_NQ_buf[iN_dimQ + iQ] += ncoef5_D_MP * eri;
                }
                K_MP_buf[iM_dimP + iP] += coef[2] * k_MP;
                K_NP_buf[iN_dimP + iP] += coef[3] * k_NP;
            }  // End of iP loop
            J_MN_buf[iM_dimN + iN] += coef[0] * j_MN;
        }  // End of iN loop
    }  // End of iM loop
    
    // Update to global array using atomic_add_f64()
    update_global_JKblk(
        dimM, dimN, dimP, dimQ, write_P,
        K_MP, K_MP_buf, K_NP, K_NP_buf,
        J_PQ, J_PQ_buf, K_MQ, K_MQ_buf, 
        K_NQ, K_NQ_buf, 1, 1
    );
}

void acc_Jmat(ACC_JKMAT_IN_PARAM)
{
    int nshell        = TinyDFT->nshell;
    int max_JKacc_buf = TinyDFT->max_JKacc_buf;
    int *shell_bf_num = TinyDFT->shell_bf_num;
    int *blk_mat_ptr  = TinyDFT->blk_mat_ptr;
    double *J_blk_mat = TinyDFT->J_blk_mat;
    double *D_blk_mat = TinyDFT->D_blk_mat;
    double *JKacc_buf = TinyDFT->JKacc_buf;
    
    // Set matrix size info
    int dimM = shell_bf_num[M];
    int dimN = shell_bf_num[N];
    int dimP = shell_bf_num[P];
    int dimQ = shell_bf_num[Q];
    
    // Set global matrix pointers
    double *J_PQ = J_blk_mat + blk_mat_ptr[P * nshell + Q];
    double *K_MP = FM_strip_buf + (blk_mat_ptr[M * nshell + P] - FM_strip_offset); 
    double *K_NP = FN_strip_buf + (blk_mat_ptr[N * nshell + P] - FN_strip_offset);
    double *K_MQ = FM_strip_buf + (blk_mat_ptr[M * nshell + Q] - FM_strip_offset);
    double *K_NQ = FN_strip_buf + (blk_mat_ptr[N * nshell + Q] - FN_strip_offset);
    
    double *D_MN = D_blk_mat + blk_mat_ptr[M * nshell + N];
    double *D_PQ = D_blk_mat + blk_mat_ptr[P * nshell + Q];
    //double *D_MP = D_blk_mat + blk_mat_ptr[M * nshell + P];
    //double *D_NP = D_blk_mat + blk_mat_ptr[N * nshell + P];
    //double *D_MQ = D_blk_mat + blk_mat_ptr[M * nshell + Q];
    //double *D_NQ = D_blk_mat + blk_mat_ptr[N * nshell + Q];
    
    // Set buffer pointer
    double *thread_buf = JKacc_buf + tid * max_JKacc_buf;
    int required_buf_size = (dimP + dimN + dimM) * dimQ + (dimP + dimN + dimM) * dimQ;
    assert(required_buf_size <= max_JKacc_buf);
    double *write_buf = thread_buf;
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * dimQ;
    double *K_NQ_buf = write_buf;  write_buf += dimN * dimQ;
    double *K_MQ_buf = write_buf;  write_buf += dimM * dimQ;

    // Reset result buffer
    //if (load_MN) memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
    if (load_P) memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
    memset(J_PQ_buf, 0, sizeof(double) * dimQ * (dimM + dimN + dimP));
    
    // Get uniqueness ERI symmetric 
    double coef[6];
    unique_integral_coef(M, N, P, Q, coef);
    
    for (int iM = 0; iM < dimM; iM++) 
    {
        int iM_dimN = iM * dimN;
        //int iM_dimP = iM * dimP;
        //int iM_dimQ = iM * dimQ;
        for (int iN = 0; iN < dimN; iN++) 
        {
            //int iN_dimP = iN * dimP;
            //int iN_dimQ = iN * dimQ;
            double coef1_D_MN = coef[1] * D_MN[iM_dimN + iN];
            double j_MN = 0.0;
            for (int iP = 0; iP < dimP; iP++) 
            {
                int iP_dimQ = iP * dimQ;
                int Ibase = dimQ * (iP + dimP * (iN + dimN * iM));
                //double ncoef4_D_NP = coef[4] * D_NP[iN_dimP + iP];
                //double ncoef5_D_MP = coef[5] * D_MP[iM_dimP + iP];
                //double k_MP = 0.0, k_NP = 0.0;
                // dimQ is small, vectorizing short loops may hurt performance 
                // since it needs horizon reduction after the loop
                for (int iQ = 0; iQ < dimQ; iQ++) 
                {
                    double eri = ERI[Ibase + iQ];
                    
                    j_MN += D_PQ[iP_dimQ + iQ] * eri;
                    //k_MP += D_NQ[iN_dimQ + iQ] * eri;
                    //k_NP += D_MQ[iM_dimQ + iQ] * eri;
                    
                    J_PQ_buf[iP_dimQ + iQ] +=  coef1_D_MN * eri;
                    //K_MQ_buf[iM_dimQ + iQ] += ncoef4_D_NP * eri;
                    //K_NQ_buf[iN_dimQ + iQ] += ncoef5_D_MP * eri;
                }
                //K_MP_buf[iM_dimP + iP] += coef[2] * k_MP;
                //K_NP_buf[iN_dimP + iP] += coef[3] * k_NP;
            }  // End of iP loop
            J_MN_buf[iM_dimN + iN] += coef[0] * j_MN;
        }  // End of iN loop
    }  // End of iM loop
    
    // Update to global array using atomic_add_f64()
    update_global_JKblk(
        dimM, dimN, dimP, dimQ, write_P,
        K_MP, K_MP_buf, K_NP, K_NP_buf,
        J_PQ, J_PQ_buf, K_MQ, K_MQ_buf, 
        K_NQ, K_NQ_buf, 1, 0
    );
}

void acc_Kmat(ACC_JKMAT_IN_PARAM)
{
    int nshell        = TinyDFT->nshell;
    int max_JKacc_buf = TinyDFT->max_JKacc_buf;
    int *shell_bf_num = TinyDFT->shell_bf_num;
    int *blk_mat_ptr  = TinyDFT->blk_mat_ptr;
    double *J_blk_mat = TinyDFT->J_blk_mat;
    double *D_blk_mat = TinyDFT->D_blk_mat;
    double *JKacc_buf = TinyDFT->JKacc_buf;
    
    // Set matrix size info
    int dimM = shell_bf_num[M];
    int dimN = shell_bf_num[N];
    int dimP = shell_bf_num[P];
    int dimQ = shell_bf_num[Q];
    
    // Set global matrix pointers
    double *J_PQ = J_blk_mat + blk_mat_ptr[P * nshell + Q];
    double *K_MP = FM_strip_buf + (blk_mat_ptr[M * nshell + P] - FM_strip_offset); 
    double *K_NP = FN_strip_buf + (blk_mat_ptr[N * nshell + P] - FN_strip_offset);
    double *K_MQ = FM_strip_buf + (blk_mat_ptr[M * nshell + Q] - FM_strip_offset);
    double *K_NQ = FN_strip_buf + (blk_mat_ptr[N * nshell + Q] - FN_strip_offset);
    
    //double *D_MN = D_blk_mat + blk_mat_ptr[M * nshell + N];
    //double *D_PQ = D_blk_mat + blk_mat_ptr[P * nshell + Q];
    double *D_MP = D_blk_mat + blk_mat_ptr[M * nshell + P];
    double *D_NP = D_blk_mat + blk_mat_ptr[N * nshell + P];
    double *D_MQ = D_blk_mat + blk_mat_ptr[M * nshell + Q];
    double *D_NQ = D_blk_mat + blk_mat_ptr[N * nshell + Q];
    
    // Set buffer pointer
    double *thread_buf = JKacc_buf + tid * max_JKacc_buf;
    int required_buf_size = (dimP + dimN + dimM) * dimQ + (dimP + dimN + dimM) * dimQ;
    assert(required_buf_size <= max_JKacc_buf);
    double *write_buf = thread_buf;
    //double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * dimQ;
    double *K_NQ_buf = write_buf;  write_buf += dimN * dimQ;
    double *K_MQ_buf = write_buf;  write_buf += dimM * dimQ;

    // Reset result buffer
    //if (load_MN) memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
    if (load_P) memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
    memset(J_PQ_buf, 0, sizeof(double) * dimQ * (dimM + dimN + dimP));
    
    // Get uniqueness ERI symmetric 
    double coef[6];
    unique_integral_coef(M, N, P, Q, coef);
    
    for (int iM = 0; iM < dimM; iM++) 
    {
        //int iM_dimN = iM * dimN;
        int iM_dimP = iM * dimP;
        int iM_dimQ = iM * dimQ;
        for (int iN = 0; iN < dimN; iN++) 
        {
            int iN_dimP = iN * dimP;
            int iN_dimQ = iN * dimQ;
            //double coef1_D_MN = coef[1] * D_MN[iM_dimN + iN];
            //double j_MN = 0.0;
            for (int iP = 0; iP < dimP; iP++) 
            {
                //int iP_dimQ = iP * dimQ;
                int Ibase = dimQ * (iP + dimP * (iN + dimN * iM));
                double ncoef4_D_NP = coef[4] * D_NP[iN_dimP + iP];
                double ncoef5_D_MP = coef[5] * D_MP[iM_dimP + iP];
                double k_MP = 0.0, k_NP = 0.0;
                // dimQ is small, vectorizing short loops may hurt performance 
                // since it needs horizon reduction after the loop
                for (int iQ = 0; iQ < dimQ; iQ++) 
                {
                    double eri = ERI[Ibase + iQ];
                    
                    //j_MN += D_PQ[iP_dimQ + iQ] * eri;
                    k_MP += D_NQ[iN_dimQ + iQ] * eri;
                    k_NP += D_MQ[iM_dimQ + iQ] * eri;
                    
                    //J_PQ_buf[iP_dimQ + iQ] +=  coef1_D_MN * eri;
                    K_MQ_buf[iM_dimQ + iQ] += ncoef4_D_NP * eri;
                    K_NQ_buf[iN_dimQ + iQ] += ncoef5_D_MP * eri;
                }
                K_MP_buf[iM_dimP + iP] += coef[2] * k_MP;
                K_NP_buf[iN_dimP + iP] += coef[3] * k_NP;
            }  // End of iP loop
            //J_MN_buf[iM_dimN + iN] += coef[0] * j_MN;
        }  // End of iN loop
    }  // End of iM loop
    
    // Update to global array using atomic_add_f64()
    update_global_JKblk(
        dimM, dimN, dimP, dimQ, write_P,
        K_MP, K_MP_buf, K_NP, K_NP_buf,
        J_PQ, J_PQ_buf, K_MQ, K_MQ_buf, 
        K_NQ, K_NQ_buf, 0, 1
    );
}

#define ACC_JKMAT_PARAM TinyDFT, tid, M, N, P_list[ipair], Q_list[ipair], \
                        ERIs + ipair * nints, load_P, write_P, \
                        FM_strip_buf, FM_strip_offset, FN_strip_buf, FN_strip_offset

void acc_JKmat_with_ket_sp_list(
    TinyDFT_p TinyDFT, int tid, int M, int N, int npair, int *P_list, int *Q_list, 
    double *ERIs, int nints, double *FM_strip_buf, double *FN_strip_buf,
    int *Mpair_flag, int *Npair_flag, int build_J, int build_K
)
{
    int nshell = TinyDFT->nshell;
    int *mat_blk_ptr = TinyDFT->blk_mat_ptr;
    int load_P, write_P, prev_P = -1;
    int FM_strip_offset = mat_blk_ptr[M * nshell];
    int FN_strip_offset = mat_blk_ptr[N * nshell];
    for (int ipair = 0; ipair < npair; ipair++)
    {
        int curr_P = P_list[ipair];
        int curr_Q = Q_list[ipair];
        if (build_K)
        {
            Mpair_flag[curr_P] = 1;
            Mpair_flag[curr_Q] = 1;
            Npair_flag[curr_P] = 1;
            Npair_flag[curr_Q] = 1;
        }
        
        load_P  = (prev_P == curr_P) ? 0 : 1;
        write_P = 0;
        if (ipair == npair - 1)
        {
            write_P = 1;
        } else {
            if (curr_P != P_list[ipair + 1]) write_P = 1;
        }
        prev_P = curr_P;
        
        if (build_J == 1 && build_K == 1) acc_JKmat(ACC_JKMAT_PARAM);
        if (build_J == 1 && build_K == 0) acc_Jmat(ACC_JKMAT_PARAM);
        if (build_J == 0 && build_K == 1) acc_Kmat(ACC_JKMAT_PARAM);
    }
}

