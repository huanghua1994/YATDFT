#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "TinySCF.h"
#include "Accum_Fock.h"

static inline void unique_integral_coef(int M, int N, int P, int Q, double *coef)
{
    int flag1 = (M == N) ? 0 : 1;
    int flag2 = (P == Q) ? 0 : 1;
    int flag3 = ((M == P) && (N == Q)) ? 0 : 1;
    int flag4 = ((flag1 == 1) && (flag2 == 1)) ? 1 : 0;
    int flag5 = ((flag1 == 1) && (flag3 == 1)) ? 1 : 0;
    int flag6 = ((flag2 == 1) && (flag3 == 1)) ? 1 : 0;
    int flag7 = ((flag4 == 1) && (flag3 == 1)) ? 1 : 0;
    coef[0] = 2.0 * (1.0   + flag1 + flag2 + flag4);  // for J_MN
    coef[1] = 2.0 * (flag3 + flag5 + flag6 + flag7);  // for J_PQ
    coef[2] = 1.0   + flag3;  // for K_MP
    coef[3] = flag1 + flag5;  // for K_NP
    coef[4] = flag2 + flag6;  // for K_MQ
    coef[5] = flag4 + flag7;  // for K_NQ
}

static inline void direct_add_vector(double *dst, double *src, int length)
{
    #pragma omp simd
    for (int i = 0; i < length; i++)
        dst[i] += src[i];
}

static inline void update_global_blocks(
    int dimM, int dimN, int dimP, int dimQ, int write_P,
    double *K_MP, double *K_MP_buf, double *K_NP, double *K_NP_buf, 
    double *J_PQ, double *J_PQ_buf, double *K_MQ, double *K_MQ_buf, 
    double *K_NQ, double *K_NQ_buf
)
{
    #ifdef BUILD_K_MAT_HF
    if (write_P)
    {
        direct_add_vector(K_MP, K_MP_buf, dimM * dimP);
        direct_add_vector(K_NP, K_NP_buf, dimN * dimP);
    }
    #endif
    
    #ifdef BUILD_J_MAT_STD
    atomic_add_vector(J_PQ, J_PQ_buf, dimP * dimQ);
    #endif
    #ifdef BUILD_K_MAT_HF
    direct_add_vector(K_MQ, K_MQ_buf, dimM * dimQ);
    direct_add_vector(K_NQ, K_NQ_buf, dimN * dimQ);
    #endif
}

void Accum_Fock(ACCUM_FOCK_IN_PARAM)
{
    // Set matrix size info
    int nbf  = TinySCF->nbasfuncs;
    int dimM = TinySCF->shell_bf_num[M];
    int dimN = TinySCF->shell_bf_num[N];
    int dimP = TinySCF->shell_bf_num[P];
    int dimQ = TinySCF->shell_bf_num[Q];
    int nshells = TinySCF->nshells;
    
    int *mat_block_ptr  = TinySCF->mat_block_ptr;
    double *D_mat_block = TinySCF->D_mat_block;
    
    // Set global matrix pointers
    double *J_PQ = TinySCF->J_mat_block + TinySCF->mat_block_ptr[P * nshells + Q];
    double *K_MP = thread_F_M_band_blocks + (mat_block_ptr[M * nshells + P] - thread_M_bank_offset); 
    double *K_NP = thread_F_N_band_blocks + (mat_block_ptr[N * nshells + P] - thread_N_bank_offset);
    double *K_MQ = thread_F_M_band_blocks + (mat_block_ptr[M * nshells + Q] - thread_M_bank_offset);
    double *K_NQ = thread_F_N_band_blocks + (mat_block_ptr[N * nshells + Q] - thread_N_bank_offset);
    
    double *D_MN = D_mat_block + mat_block_ptr[M * nshells + N];
    double *D_PQ = D_mat_block + mat_block_ptr[P * nshells + Q];
    double *D_MP = D_mat_block + mat_block_ptr[M * nshells + P];
    double *D_NP = D_mat_block + mat_block_ptr[N * nshells + P];
    double *D_MQ = D_mat_block + mat_block_ptr[M * nshells + Q];
    double *D_NQ = D_mat_block + mat_block_ptr[N * nshells + Q];
    
    // Set buffer pointer
    double *thread_buf = TinySCF->Accum_Fock_buf + tid * TinySCF->max_buf_size;
    int required_buf_size = (dimP + dimN + dimM) * dimQ + (dimP + dimN + dimM) * dimQ;
    assert(required_buf_size <= TinySCF->max_buf_size);
    double *write_buf = thread_buf;
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * dimQ;
    double *K_NQ_buf = write_buf;  write_buf += dimN * dimQ;
    double *K_MQ_buf = write_buf;  write_buf += dimM * dimQ;

    // Reset result buffer
    //if (load_MN) memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
    if (load_P)  memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
    memset(J_PQ_buf, 0, sizeof(double) * dimQ * (dimM + dimN + dimP));
    
    // Get uniqueness ERI symmetric 
    double coef[7];
    unique_integral_coef(M, N, P, Q, coef);
    
    for (int iM = 0; iM < dimM; iM++) 
    {
        int iM_dimP = iM * dimP;
        int iM_dimN = iM * dimN;
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
                double ncoef4_D_NP = -coef[4] * D_NP[iN_dimP + iP];
                double ncoef5_D_MP = -coef[5] * D_MP[iM_dimP + iP];
                double k_MP = 0.0, k_NP = 0.0;
                // dimQ is small, vectorizing short loops may hurt performance 
                // since it needs horizon reduction after the loop
                for (int iQ = 0; iQ < dimQ; iQ++) 
                {
                    double I = ERI[Ibase + iQ];
                    
                    #ifdef BUILD_J_MAT_STD
                    j_MN += D_PQ[iP_dimQ + iQ] * I;
                    #endif
                    #ifdef BUILD_K_MAT_HF
                    k_MP -= D_NQ[iN_dimQ + iQ] * I;
                    k_NP -= D_MQ[iM_dimQ + iQ] * I;
                    #endif
                    
                    #ifdef BUILD_J_MAT_STD
                    J_PQ_buf[iP_dimQ + iQ] +=  coef1_D_MN * I;
                    #endif
                    #ifdef BUILD_K_MAT_HF
                    K_MQ_buf[iM_dimQ + iQ] += ncoef4_D_NP * I;
                    K_NQ_buf[iN_dimQ + iQ] += ncoef5_D_MP * I;
                    #endif
                }
                #ifdef BUILD_K_MAT_HF
                K_MP_buf[iM_dimP + iP] += coef[2] * k_MP;
                K_NP_buf[iN_dimP + iP] += coef[3] * k_NP;
                #endif
            }  // for (int iP = 0; iP < dimP; iP++) 
            #ifdef BUILD_J_MAT_STD
            J_MN_buf[iM_dimN + iN] += coef[0] * j_MN;
            #endif
        } // for (int iN = 0; iN < dimN; iN++) 
    } // for (int iM = 0; iM < dimM; iM++) 
    
    // Update to global array using atomic_add_f64()
    update_global_blocks(
        dimM, dimN, dimP, dimQ, write_P,
        K_MP, K_MP_buf, K_NP, K_NP_buf,
        J_PQ, J_PQ_buf, K_MQ, K_MQ_buf, K_NQ, K_NQ_buf
    );
}

// ----- Specialized implementations of Accum_Fock with different dimQ -----
// ----- We don't have function template in C, so we have to copy them -----

void Accum_Fock_dimQ1(ACCUM_FOCK_IN_PARAM)
{
    // Set matrix size info
    int nbf  = TinySCF->nbasfuncs;
    int dimM = TinySCF->shell_bf_num[M];
    int dimN = TinySCF->shell_bf_num[N];
    int dimP = TinySCF->shell_bf_num[P];
    int dimQ = TinySCF->shell_bf_num[Q];
    int nshells = TinySCF->nshells;
    
    int *mat_block_ptr  = TinySCF->mat_block_ptr;
    double *D_mat_block = TinySCF->D_mat_block;
    
    // Set global matrix pointers
    double *J_PQ = TinySCF->J_mat_block + TinySCF->mat_block_ptr[P * nshells + Q];
    double *K_MP = thread_F_M_band_blocks + (mat_block_ptr[M * nshells + P] - thread_M_bank_offset); 
    double *K_NP = thread_F_N_band_blocks + (mat_block_ptr[N * nshells + P] - thread_N_bank_offset);
    double *K_MQ = thread_F_M_band_blocks + (mat_block_ptr[M * nshells + Q] - thread_M_bank_offset);
    double *K_NQ = thread_F_N_band_blocks + (mat_block_ptr[N * nshells + Q] - thread_N_bank_offset);
    
    double *D_MN = D_mat_block + mat_block_ptr[M * nshells + N];
    double *D_PQ = D_mat_block + mat_block_ptr[P * nshells + Q];
    double *D_MP = D_mat_block + mat_block_ptr[M * nshells + P];
    double *D_NP = D_mat_block + mat_block_ptr[N * nshells + P];
    double *D_MQ = D_mat_block + mat_block_ptr[M * nshells + Q];
    double *D_NQ = D_mat_block + mat_block_ptr[N * nshells + Q];
    
    // Set buffer pointer
    double *thread_buf = TinySCF->Accum_Fock_buf + tid * TinySCF->max_buf_size;
    int required_buf_size = (dimP + dimN + dimM) * dimQ + (dimP + dimN + dimM) * dimQ;
    assert(required_buf_size <= TinySCF->max_buf_size);
    double *write_buf = thread_buf;
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * dimQ;
    double *K_NQ_buf = write_buf;  write_buf += dimN * dimQ;
    double *K_MQ_buf = write_buf;  write_buf += dimM * dimQ;

    // Reset result buffer
    if (load_P)  memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
    memset(J_PQ_buf, 0, sizeof(double) * dimQ * (dimM + dimN + dimP));
    
    // Get uniqueness ERI symmetric 
    double coef[7];
    unique_integral_coef(M, N, P, Q, coef);
    
    for (int iM = 0; iM < dimM; iM++) 
    {
        int iM_dimN = iM * dimN;
        for (int iN = 0; iN < dimN; iN++) 
        {
            int iM_dimP = iM * dimP;
            int iN_dimP = iN * dimP;
            double coef1_D_MN = coef[1] * D_MN[iM_dimN + iN];
            double j_MN = 0.0, k_MQ = 0.0, k_NQ = 0.0;
            int Ibase = dimP * (iN + dimN * iM);
            for (int iP = 0; iP < dimP; iP++) 
            {
                double ncoef4_D_NP = -coef[4] * D_NP[iN_dimP + iP];
                double ncoef5_D_MP = -coef[5] * D_MP[iM_dimP + iP];

                double I = ERI[Ibase + iP];
                
                #ifdef BUILD_J_MAT_STD
                j_MN += D_PQ[iP] * I;
                #endif
                #ifdef BUILD_K_MAT_HF
                k_MQ += ncoef4_D_NP * I;
                k_NQ += ncoef5_D_MP * I;
                #endif
                
                #ifdef BUILD_J_MAT_STD
                J_PQ_buf[iP] += coef1_D_MN * I;
                #endif
                #ifdef BUILD_K_MAT_HF
                K_MP_buf[iM_dimP + iP] -= coef[2] * D_NQ[iN] * I;
                K_NP_buf[iN_dimP + iP] -= coef[3] * D_MQ[iM] * I;
                #endif
            }  // for (int iP = 0; iP < dimP; iP++)  
            #ifdef BUILD_K_MAT_HF
            K_MQ_buf[iM] += k_MQ;
            K_NQ_buf[iN] += k_NQ;
            #endif
            #ifdef BUILD_J_MAT_STD
            J_MN_buf[iM_dimN + iN] += coef[0] * j_MN;
            #endif
        }  // for (int iN = 0; iN < dimN; iN++) 
    }  // for (int iM = 0; iM < dimM; iM++) 
    
    // Update to global array using atomic_add_f64()
    update_global_blocks(
        dimM, dimN, dimP, dimQ, write_P,
        K_MP, K_MP_buf, K_NP, K_NP_buf,    J_PQ, J_PQ_buf, 
        K_MQ, K_MQ_buf, K_NQ, K_NQ_buf
    );
}

void Accum_Fock_dimQ3(ACCUM_FOCK_IN_PARAM)
{
    // Set matrix size info
    int nbf  = TinySCF->nbasfuncs;
    int dimM = TinySCF->shell_bf_num[M];
    int dimN = TinySCF->shell_bf_num[N];
    int dimP = TinySCF->shell_bf_num[P];
    int dimQ = 3; // TinySCF->shell_bf_num[Q];
    int nshells = TinySCF->nshells;
    
    int *mat_block_ptr  = TinySCF->mat_block_ptr;
    double *D_mat_block = TinySCF->D_mat_block;
    
    // Set global matrix pointers
    double *J_PQ = TinySCF->J_mat_block + TinySCF->mat_block_ptr[P * nshells + Q];
    double *K_MP = thread_F_M_band_blocks + (mat_block_ptr[M * nshells + P] - thread_M_bank_offset); 
    double *K_NP = thread_F_N_band_blocks + (mat_block_ptr[N * nshells + P] - thread_N_bank_offset);
    double *K_MQ = thread_F_M_band_blocks + (mat_block_ptr[M * nshells + Q] - thread_M_bank_offset);
    double *K_NQ = thread_F_N_band_blocks + (mat_block_ptr[N * nshells + Q] - thread_N_bank_offset);
    
    double *D_MN = D_mat_block + mat_block_ptr[M * nshells + N];
    double *D_PQ = D_mat_block + mat_block_ptr[P * nshells + Q];
    double *D_MP = D_mat_block + mat_block_ptr[M * nshells + P];
    double *D_NP = D_mat_block + mat_block_ptr[N * nshells + P];
    double *D_MQ = D_mat_block + mat_block_ptr[M * nshells + Q];
    double *D_NQ = D_mat_block + mat_block_ptr[N * nshells + Q];
    
    // Set buffer pointer
    double *thread_buf = TinySCF->Accum_Fock_buf + tid * TinySCF->max_buf_size;
    int required_buf_size = (dimP + dimN + dimM) * 3 + (dimP + dimN + dimM) * 3;
    assert(required_buf_size <= TinySCF->max_buf_size);
    double *write_buf = thread_buf;
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * 3;
    double *K_NQ_buf = write_buf;  write_buf += dimN * 3;
    double *K_MQ_buf = write_buf;  write_buf += dimM * 3;

    // Reset result buffer
    if (load_P)  memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
    memset(J_PQ_buf, 0, sizeof(double) * 3 * (dimM + dimN + dimP));
    
    // Get uniqueness ERI symmetric 
    double coef[7];
    unique_integral_coef(M, N, P, Q, coef);
    
    for (int iM = 0; iM < dimM; iM++) 
    {
        int iM_dimP = iM * dimP;
        int iM_dimN = iM * dimN;
        int iM_dimQ = iM * 3;
        for (int iN = 0; iN < dimN; iN++) 
        {
            int iN_dimP = iN * dimP;
            int iN_dimQ = iN * 3;
            double coef1_D_MN = coef[1] * D_MN[iM_dimN + iN];
            double j_MN = 0.0;
            for (int iP = 0; iP < dimP; iP++) 
            {
                int iP_dimQ = iP * 3;
                int Ibase = 3 * (iP + dimP * (iN + dimN * iM));
                double ncoef4_D_NP = -coef[4] * D_NP[iN_dimP + iP];
                double ncoef5_D_MP = -coef[5] * D_MP[iM_dimP + iP];
                double k_MP = 0.0, k_NP = 0.0;
                
                
                for (int iQ = 0; iQ < 3; iQ++) 
                {
                    double I = ERI[Ibase + iQ];
                    
                    #ifdef BUILD_J_MAT_STD
                    j_MN += D_PQ[iP_dimQ + iQ] * I;
                    #endif
                    #ifdef BUILD_K_MAT_HF
                    k_MP -= D_NQ[iN_dimQ + iQ] * I;
                    k_NP -= D_MQ[iM_dimQ + iQ] * I;
                    #endif
                    
                    #ifdef BUILD_J_MAT_STD
                    J_PQ_buf[iP_dimQ + iQ] +=  coef1_D_MN * I;
                    #endif
                    #ifdef BUILD_K_MAT_HF
                    K_MQ_buf[iM_dimQ + iQ] += ncoef4_D_NP * I;
                    K_NQ_buf[iN_dimQ + iQ] += ncoef5_D_MP * I;
                    #endif
                }
                #ifdef BUILD_K_MAT_HF
                K_MP_buf[iM_dimP + iP] += coef[2] * k_MP;
                K_NP_buf[iN_dimP + iP] += coef[3] * k_NP;
                #endif
            }  // for (int iP = 0; iP < dimP; iP++) 
            #ifdef BUILD_J_MAT_STD
            J_MN_buf[iM_dimN + iN] += coef[0] * j_MN;
            #endif
        }  // for (int iN = 0; iN < dimN; iN++) 
    }  // for (int iM = 0; iM < dimM; iM++) 
    
    // Update to global array using atomic_add_f64()
    update_global_blocks(
        dimM, dimN, dimP, dimQ, write_P,
        K_MP, K_MP_buf, K_NP, K_NP_buf,    J_PQ, J_PQ_buf, 
        K_MQ, K_MQ_buf, K_NQ, K_NQ_buf
    );
}

void Accum_Fock_dimQ6(ACCUM_FOCK_IN_PARAM)
{
    // Set matrix size info
    int nbf  = TinySCF->nbasfuncs;
    int dimM = TinySCF->shell_bf_num[M];
    int dimN = TinySCF->shell_bf_num[N];
    int dimP = TinySCF->shell_bf_num[P];
    int dimQ = 6; // TinySCF->shell_bf_num[Q];
    int nshells = TinySCF->nshells;
    
    int *mat_block_ptr  = TinySCF->mat_block_ptr;
    double *D_mat_block = TinySCF->D_mat_block;
    
    // Set global matrix pointers
    double *J_PQ = TinySCF->J_mat_block + TinySCF->mat_block_ptr[P * nshells + Q];
    double *K_MP = thread_F_M_band_blocks + (mat_block_ptr[M * nshells + P] - thread_M_bank_offset); 
    double *K_NP = thread_F_N_band_blocks + (mat_block_ptr[N * nshells + P] - thread_N_bank_offset);
    double *K_MQ = thread_F_M_band_blocks + (mat_block_ptr[M * nshells + Q] - thread_M_bank_offset);
    double *K_NQ = thread_F_N_band_blocks + (mat_block_ptr[N * nshells + Q] - thread_N_bank_offset);
    
    double *D_MN = D_mat_block + mat_block_ptr[M * nshells + N];
    double *D_PQ = D_mat_block + mat_block_ptr[P * nshells + Q];
    double *D_MP = D_mat_block + mat_block_ptr[M * nshells + P];
    double *D_NP = D_mat_block + mat_block_ptr[N * nshells + P];
    double *D_MQ = D_mat_block + mat_block_ptr[M * nshells + Q];
    double *D_NQ = D_mat_block + mat_block_ptr[N * nshells + Q];
    
    // Set buffer pointer
    double *thread_buf = TinySCF->Accum_Fock_buf + tid * TinySCF->max_buf_size;
    int required_buf_size = (dimP + dimN + dimM) * 6 + (dimP + dimN + dimM) * 6;
    assert(required_buf_size <= TinySCF->max_buf_size);
    double *write_buf = thread_buf;
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * 6;
    double *K_NQ_buf = write_buf;  write_buf += dimN * 6;
    double *K_MQ_buf = write_buf;  write_buf += dimM * 6;

    // Reset result buffer
    if (load_P)  memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
    memset(J_PQ_buf, 0, sizeof(double) * 6 * (dimM + dimN + dimP));
    
    // Get uniqueness ERI symmetric 
    double coef[7];
    unique_integral_coef(M, N, P, Q, coef);
    
    for (int iM = 0; iM < dimM; iM++) 
    {
        int iM_dimP = iM * dimP;
        int iM_dimN = iM * dimN;
        int iM_dimQ = iM * 6;
        for (int iN = 0; iN < dimN; iN++) 
        {
            int iN_dimP = iN * dimP;
            int iN_dimQ = iN * 6;
            double coef1_D_MN = coef[1] * D_MN[iM_dimN + iN];
            double j_MN = 0.0;
            for (int iP = 0; iP < dimP; iP++) 
            {
                int iP_dimQ = iP * 6;
                int Ibase = 6 * (iP + dimP * (iN + dimN * iM));
                double ncoef4_D_NP = -coef[4] * D_NP[iN_dimP + iP];
                double ncoef5_D_MP = -coef[5] * D_MP[iM_dimP + iP];
                double k_MP = 0.0, k_NP = 0.0;
                
                for (int iQ = 0; iQ < 6; iQ++) 
                {
                    double I = ERI[Ibase + iQ];
                    
                    #ifdef BUILD_J_MAT_STD
                    j_MN += D_PQ[iP_dimQ + iQ] * I;
                    #endif
                    #ifdef BUILD_K_MAT_HF
                    k_MP -= D_NQ[iN_dimQ + iQ] * I;
                    k_NP -= D_MQ[iM_dimQ + iQ] * I;
                    #endif
                    
                    #ifdef BUILD_J_MAT_STD
                    J_PQ_buf[iP_dimQ + iQ] +=  coef1_D_MN * I;
                    #endif
                    #ifdef BUILD_K_MAT_HF
                    K_MQ_buf[iM_dimQ + iQ] += ncoef4_D_NP * I;
                    K_NQ_buf[iN_dimQ + iQ] += ncoef5_D_MP * I;
                    #endif
                }
                #ifdef BUILD_K_MAT_HF
                K_MP_buf[iM_dimP + iP] += coef[2] * k_MP;
                K_NP_buf[iN_dimP + iP] += coef[3] * k_NP;
                #endif
            }  // for (int iP = 0; iP < dimP; iP++) 
            #ifdef BUILD_J_MAT_STD
            J_MN_buf[iM_dimN + iN] += coef[0] * j_MN;
            #endif
        } // for (int iN = 0; iN < dimN; iN++) 
    } // for (int iM = 0; iM < dimM; iM++) 
    
    // Update to global array using atomic_add_f64()
    update_global_blocks(
        dimM, dimN, dimP, dimQ, write_P,
        K_MP, K_MP_buf, K_NP, K_NP_buf,    J_PQ, J_PQ_buf, 
        K_MQ, K_MQ_buf, K_NQ, K_NQ_buf
    );
}

void Accum_Fock_dimQ10(ACCUM_FOCK_IN_PARAM)
{
    // Set matrix size info
    int nbf  = TinySCF->nbasfuncs;
    int dimM = TinySCF->shell_bf_num[M];
    int dimN = TinySCF->shell_bf_num[N];
    int dimP = TinySCF->shell_bf_num[P];
    int dimQ = 10; // TinySCF->shell_bf_num[Q];
    int nshells = TinySCF->nshells;
    
    int *mat_block_ptr  = TinySCF->mat_block_ptr;
    double *D_mat_block = TinySCF->D_mat_block;
    
    // Set global matrix pointers
    double *J_PQ = TinySCF->J_mat_block + TinySCF->mat_block_ptr[P * nshells + Q];
    double *K_MP = thread_F_M_band_blocks + (mat_block_ptr[M * nshells + P] - thread_M_bank_offset); 
    double *K_NP = thread_F_N_band_blocks + (mat_block_ptr[N * nshells + P] - thread_N_bank_offset);
    double *K_MQ = thread_F_M_band_blocks + (mat_block_ptr[M * nshells + Q] - thread_M_bank_offset);
    double *K_NQ = thread_F_N_band_blocks + (mat_block_ptr[N * nshells + Q] - thread_N_bank_offset);
    
    double *D_MN = D_mat_block + mat_block_ptr[M * nshells + N];
    double *D_PQ = D_mat_block + mat_block_ptr[P * nshells + Q];
    double *D_MP = D_mat_block + mat_block_ptr[M * nshells + P];
    double *D_NP = D_mat_block + mat_block_ptr[N * nshells + P];
    double *D_MQ = D_mat_block + mat_block_ptr[M * nshells + Q];
    double *D_NQ = D_mat_block + mat_block_ptr[N * nshells + Q];
    
    // Set buffer pointer
    double *thread_buf = TinySCF->Accum_Fock_buf + tid * TinySCF->max_buf_size;
    int required_buf_size = (dimP + dimN + dimM) * 10 + (dimP + dimN + dimM) * 10;
    assert(required_buf_size <= TinySCF->max_buf_size);
    double *write_buf = thread_buf;
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * 10;
    double *K_NQ_buf = write_buf;  write_buf += dimN * 10;
    double *K_MQ_buf = write_buf;  write_buf += dimM * 10;

    // Reset result buffer
    if (load_P)  memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
    memset(J_PQ_buf, 0, sizeof(double) * 10 * (dimM + dimN + dimP));
    
    // Get uniqueness ERI symmetric 
    double coef[7];
    unique_integral_coef(M, N, P, Q, coef);
    
    for (int iM = 0; iM < dimM; iM++) 
    {
        int iM_dimP = iM * dimP;
        int iM_dimN = iM * dimN;
        int iM_dimQ = iM * 10;
        for (int iN = 0; iN < dimN; iN++) 
        {
            int iN_dimP = iN * dimP;
            int iN_dimQ = iN * 10;
            double coef1_D_MN = coef[1] * D_MN[iM_dimN + iN];
            double j_MN = 0.0;
            for (int iP = 0; iP < dimP; iP++) 
            {
                int iP_dimQ = iP * 10;
                int Ibase = 10 * (iP + dimP * (iN + dimN * iM));
                double ncoef4_D_NP = -coef[4] * D_NP[iN_dimP + iP];
                double ncoef5_D_MP = -coef[5] * D_MP[iM_dimP + iP];
                double k_MP = 0.0, k_NP = 0.0;
                
                for (int iQ = 0; iQ < 10; iQ++) 
                {
                    double I = ERI[Ibase + iQ];
                    
                    #ifdef BUILD_J_MAT_STD
                    j_MN += D_PQ[iP_dimQ + iQ] * I;
                    #endif
                    #ifdef BUILD_K_MAT_HF
                    k_MP -= D_NQ[iN_dimQ + iQ] * I;
                    k_NP -= D_MQ[iM_dimQ + iQ] * I;
                    #endif
                    
                    #ifdef BUILD_J_MAT_STD
                    J_PQ_buf[iP_dimQ + iQ] +=  coef1_D_MN * I;
                    #endif
                    #ifdef BUILD_K_MAT_HF
                    K_MQ_buf[iM_dimQ + iQ] += ncoef4_D_NP * I;
                    K_NQ_buf[iN_dimQ + iQ] += ncoef5_D_MP * I;
                    #endif
                }
                #ifdef BUILD_K_MAT_HF
                K_MP_buf[iM_dimP + iP] += coef[2] * k_MP;
                K_NP_buf[iN_dimP + iP] += coef[3] * k_NP;
                #endif
            }  // for (int iP = 0; iP < dimP; iP++) 
            #ifdef BUILD_J_MAT_STD
            J_MN_buf[iM_dimN + iN] += coef[0] * j_MN;
            #endif
        } // for (int iN = 0; iN < dimN; iN++) 
    } // for (int iM = 0; iM < dimM; iM++) 
    
    // Update to global array using atomic_add_f64()
    update_global_blocks(
        dimM, dimN, dimP, dimQ, write_P,
        K_MP, K_MP_buf, K_NP, K_NP_buf,    J_PQ, J_PQ_buf, 
        K_MQ, K_MQ_buf, K_NQ, K_NQ_buf
    );
}

void Accum_Fock_dimQ15(ACCUM_FOCK_IN_PARAM)
{
    // Set matrix size info
    int nbf  = TinySCF->nbasfuncs;
    int dimM = TinySCF->shell_bf_num[M];
    int dimN = TinySCF->shell_bf_num[N];
    int dimP = TinySCF->shell_bf_num[P];
    int dimQ = 15; // TinySCF->shell_bf_num[Q];
    int nshells = TinySCF->nshells;
    
    int *mat_block_ptr  = TinySCF->mat_block_ptr;
    double *D_mat_block = TinySCF->D_mat_block;
    
    // Set global matrix pointers
    double *J_PQ = TinySCF->J_mat_block + TinySCF->mat_block_ptr[P * nshells + Q];
    double *K_MP = thread_F_M_band_blocks + (mat_block_ptr[M * nshells + P] - thread_M_bank_offset); 
    double *K_NP = thread_F_N_band_blocks + (mat_block_ptr[N * nshells + P] - thread_N_bank_offset);
    double *K_MQ = thread_F_M_band_blocks + (mat_block_ptr[M * nshells + Q] - thread_M_bank_offset);
    double *K_NQ = thread_F_N_band_blocks + (mat_block_ptr[N * nshells + Q] - thread_N_bank_offset);
    
    double *D_MN = D_mat_block + mat_block_ptr[M * nshells + N];
    double *D_PQ = D_mat_block + mat_block_ptr[P * nshells + Q];
    double *D_MP = D_mat_block + mat_block_ptr[M * nshells + P];
    double *D_NP = D_mat_block + mat_block_ptr[N * nshells + P];
    double *D_MQ = D_mat_block + mat_block_ptr[M * nshells + Q];
    double *D_NQ = D_mat_block + mat_block_ptr[N * nshells + Q];
    
    // Set buffer pointer
    double *thread_buf = TinySCF->Accum_Fock_buf + tid * TinySCF->max_buf_size;
    int required_buf_size = (dimP + dimN + dimM) * 15 + (dimP + dimN + dimM) * 15;
    assert(required_buf_size <= TinySCF->max_buf_size);
    double *write_buf = thread_buf;
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * 15;
    double *K_NQ_buf = write_buf;  write_buf += dimN * 15;
    double *K_MQ_buf = write_buf;  write_buf += dimM * 15;

    // Reset result buffer
    if (load_P)  memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
    memset(J_PQ_buf, 0, sizeof(double) * 15 * (dimM + dimN + dimP));
    
    // Get uniqueness ERI symmetric 
    double coef[7];
    unique_integral_coef(M, N, P, Q, coef);
    
    for (int iM = 0; iM < dimM; iM++) 
    {
        int iM_dimP = iM * dimP;
        int iM_dimN = iM * dimN;
        int iM_dimQ = iM * 15;
        for (int iN = 0; iN < dimN; iN++) 
        {
            int iN_dimP = iN * dimP;
            int iN_dimQ = iN * 15;
            double coef1_D_MN = coef[1] * D_MN[iM_dimN + iN];
            double j_MN = 0.0;
            for (int iP = 0; iP < dimP; iP++) 
            {
                int iP_dimQ = iP * 15;
                int Ibase = dimQ * (iP + dimP * (iN + dimN * iM));
                double ncoef4_D_NP = -coef[4] * D_NP[iN_dimP + iP];
                double ncoef5_D_MP = -coef[5] * D_MP[iM_dimP + iP];
                double k_MP = 0.0, k_NP = 0.0;
                
                #pragma omp simd
                for (int iQ = 0; iQ < 15; iQ++) 
                {
                    double I = ERI[Ibase + iQ];
                    
                    #ifdef BUILD_J_MAT_STD
                    j_MN += D_PQ[iP_dimQ + iQ] * I;
                    #endif
                    #ifdef BUILD_K_MAT_HF
                    k_MP -= D_NQ[iN_dimQ + iQ] * I;
                    k_NP -= D_MQ[iM_dimQ + iQ] * I;
                    #endif
                    
                    #ifdef BUILD_J_MAT_STD
                    J_PQ_buf[iP_dimQ + iQ] +=  coef1_D_MN * I;
                    #endif
                    #ifdef BUILD_K_MAT_HF
                    K_MQ_buf[iM_dimQ + iQ] += ncoef4_D_NP * I;
                    K_NQ_buf[iN_dimQ + iQ] += ncoef5_D_MP * I;
                    #endif
                }
                #ifdef BUILD_K_MAT_HF
                K_MP_buf[iM_dimP + iP] += coef[2] * k_MP;
                K_NP_buf[iN_dimP + iP] += coef[3] * k_NP;
                #endif
            }  // for (int iP = 0; iP < dimP; iP++) 
            #ifdef BUILD_J_MAT_STD
            J_MN_buf[iM_dimN + iN] += coef[0] * j_MN;
            #endif
        } // for (int iN = 0; iN < dimN; iN++) 
    } // for (int iM = 0; iM < dimM; iM++) 
    
    // Update to global array using atomic_add_f64()
    update_global_blocks(
        dimM, dimN, dimP, dimQ, write_P,
        K_MP, K_MP_buf, K_NP, K_NP_buf,    J_PQ, J_PQ_buf, 
        K_MQ, K_MQ_buf, K_NQ, K_NQ_buf
    );
}

void Accum_Fock_1111(ACCUM_FOCK_IN_PARAM)
{
    // Set matrix size info
    int nshells = TinySCF->nshells;
    
    int *mat_block_ptr  = TinySCF->mat_block_ptr;
    double *D_mat_block = TinySCF->D_mat_block;
    
    // Set global matrix pointers
    double *J_PQ = TinySCF->J_mat_block + TinySCF->mat_block_ptr[P * nshells + Q];
    double *K_MP = thread_F_M_band_blocks + (mat_block_ptr[M * nshells + P] - thread_M_bank_offset); 
    double *K_NP = thread_F_N_band_blocks + (mat_block_ptr[N * nshells + P] - thread_N_bank_offset);
    double *K_MQ = thread_F_M_band_blocks + (mat_block_ptr[M * nshells + Q] - thread_M_bank_offset);
    double *K_NQ = thread_F_N_band_blocks + (mat_block_ptr[N * nshells + Q] - thread_N_bank_offset);
    
    double *D_MN = D_mat_block + mat_block_ptr[M * nshells + N];
    double *D_PQ = D_mat_block + mat_block_ptr[P * nshells + Q];
    double *D_MP = D_mat_block + mat_block_ptr[M * nshells + P];
    double *D_NP = D_mat_block + mat_block_ptr[N * nshells + P];
    double *D_MQ = D_mat_block + mat_block_ptr[M * nshells + Q];
    double *D_NQ = D_mat_block + mat_block_ptr[N * nshells + Q];

    // Get uniqueness ERI symmetric 
    double coef[7];
    unique_integral_coef(M, N, P, Q, coef);
    
    double I = ERI[0];
    double *J_MN_buf = TinySCF->Accum_Fock_buf + tid * TinySCF->max_buf_size;
    
    #ifdef BUILD_J_MAT_STD
    double vMN =  coef[0] * D_PQ[0] * I;
    double vPQ =  coef[1] * D_MN[0] * I;
    #endif
    #ifdef BUILD_K_MAT_HF
    double vMP = -coef[2] * D_NQ[0] * I;
    double vNP = -coef[3] * D_MQ[0] * I;
    double vMQ = -coef[4] * D_NP[0] * I;
    double vNQ = -coef[5] * D_MP[0] * I;
    #endif
    
    #ifdef BUILD_J_MAT_STD
    J_MN_buf[0] += vMN;
    atomic_add_f64(&J_PQ[0], vPQ);
    #endif
    #ifdef BUILD_K_MAT_HF
    K_MP[0] += vMP;
    K_NP[0] += vNP;
    K_MQ[0] += vMQ;
    K_NQ[0] += vNQ;
    #endif
}
