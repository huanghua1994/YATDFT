#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#include <mkl.h>

#include "utils.h"
#include "TinySCF.h"
#include "build_Fock.h"

void TinySCF_DIIS(TinySCF_t TinySCF)
{
    double *S_mat    = TinySCF->S_mat;
    double *F_mat    = TinySCF->F_mat;
    double *D_mat    = TinySCF->D_mat;
    double *X_mat    = TinySCF->X_mat;
    double *F0_mat   = TinySCF->F0_mat;
    double *R_mat    = TinySCF->R_mat;
    double *B_mat    = TinySCF->B_mat;
    double *FDS_mat  = TinySCF->FDS_mat;
    double *DIIS_rhs = TinySCF->DIIS_rhs;
    double *tmp_mat  = TinySCF->tmp_mat;
    int    *ipiv     = TinySCF->DIIS_ipiv;
    int mat_size     = TinySCF->mat_size;
    int mat_mem_size = DBL_SIZE * mat_size;
    int ldB = MAX_DIIS + 1;
    int nbf = TinySCF->nbasfuncs;
    
    if (TinySCF->iter <= 1)
    {
        // F = X^T * F * X
        // Use tmp_mat to store X^T * F
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nbf, nbf, nbf, 
                    1.0, X_mat, nbf, F_mat, nbf, 0.0, tmp_mat, nbf);
        // Use F_mat to store X^T * F * X
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf, 
                    1.0, tmp_mat, nbf, X_mat, nbf, 0.0, F_mat, nbf);
        return;
    }
    
    int DIIS_idx;   // Which historic F matrix will be replaced
    if (TinySCF->DIIS_len < MAX_DIIS)
    {
        DIIS_idx = TinySCF->DIIS_len;
        TinySCF->DIIS_len++;
    } else {
        DIIS_idx = TinySCF->DIIS_bmax_id;
    }
    
    // FDS = F * D * S;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf,
                1.0, F_mat, nbf, D_mat, nbf, 0.0, tmp_mat, nbf);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf,
                1.0, tmp_mat, nbf, S_mat, nbf, 0.0, FDS_mat, nbf);
    
    // Residual = X^T * (FDS - FDS^T) * X
    // Use tmp_mat to store FDS - FDS^T
    mkl_domatadd('R', 'N', 'T', nbf, nbf, 1.0, FDS_mat, nbf, -1.0, FDS_mat, nbf, tmp_mat, nbf);
    // Use FDS_mat to store X^T * (FDS - FDS^T)
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nbf, nbf, nbf,
                1.0, X_mat, nbf, tmp_mat, nbf, 0.0, FDS_mat, nbf);
    // Use tmp_mat to store X^T * (FDS - FDS^T) * X
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf,
                1.0, FDS_mat, nbf, X_mat, nbf, 0.0, tmp_mat, nbf);
    
    // In my MATLAB code, F_mat and its residual are treated as column vectors
    // For performance, we treat them as row vectors here
    
    // R(:, DIIS_idx) = X^T * (FDS - FDS^T) * X
    // B(i, j) = R(:, i) * R(:, j)
    // DIIS_rhs is not used yet, use it to store dot product results
    double *DIIS_dot = DIIS_rhs; 
    memset(DIIS_dot, 0, DBL_SIZE * (MAX_DIIS + 1));
    memcpy(R_mat + mat_size * DIIS_idx, tmp_mat, mat_mem_size);
    double *Ri = R_mat + mat_size * DIIS_idx;
    for (int j = 0; j < TinySCF->DIIS_len; j++)
    {
        double *Rj = R_mat + mat_size * j;
        DIIS_dot[j] = cblas_ddot(mat_size, Ri, 1, Rj, 1);
    }
    
    // Construct symmetric B
    // B(DIIS_idx, 1 : DIIS_len) = DIIS_dot(1 : DIIS_idx);
    // B(1 : DIIS_len, DIIS_idx) = DIIS_dot(1 : DIIS_idx);
    for (int i = 0; i < TinySCF->DIIS_len; i++)
    {
        B_mat[DIIS_idx * ldB + i] = DIIS_dot[i];
        B_mat[i * ldB + DIIS_idx] = DIIS_dot[i];
    }
    
    // Pick an old F that its residual has the largest 2-norm
    for (int i = 0; i < TinySCF->DIIS_len; i++)
    {
        if (B_mat[i * ldB + i] > TinySCF->DIIS_bmax)
        {
            TinySCF->DIIS_bmax    = B_mat[i * ldB + i];
            TinySCF->DIIS_bmax_id = i;
        }
    }
    
    // F := X^T * F * X, F0(:, DIIS_idx) = F
    // Use tmp_mat to store X^T * F
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nbf, nbf, nbf, 
                1.0, X_mat, nbf, F_mat, nbf, 0.0, tmp_mat, nbf);
    // Use F_mat to store X^T * F * X
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf, 
                1.0, tmp_mat, nbf, X_mat, nbf, 0.0, F_mat, nbf);
    // Copy to F0
    memcpy(F0_mat + mat_size * DIIS_idx, F_mat, mat_mem_size);
    
    // Solve the linear system 
    memset(DIIS_rhs, 0, DBL_SIZE * (MAX_DIIS + 1));
    DIIS_rhs[TinySCF->DIIS_len] = -1;
    // Copy B_mat to tmp_mat, since LAPACKE_dgesv will overwrite the input matrix
    memcpy(tmp_mat, B_mat, DBL_SIZE * ldB * ldB);  
    LAPACKE_dgesv(LAPACK_ROW_MAJOR, TinySCF->DIIS_len + 1, 1, tmp_mat, ldB, ipiv, DIIS_rhs, 1);
    
    // Form new X^T * F * X
    memset(F_mat, 0, mat_mem_size);
    for (int i = 0; i < TinySCF->DIIS_len; i++)
        cblas_daxpy(mat_size, DIIS_rhs[i], F0_mat + i * mat_size, 1, F_mat, 1);
}
