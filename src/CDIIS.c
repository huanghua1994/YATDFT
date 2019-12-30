#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#include <mkl.h>

#include "TinyDFT_typedef.h"
#include "CDIIS.h"
#include "utils.h"

void TinyDFT_CDIIS(TinyDFT_t TinyDFT, const double *X_mat, const double *S_mat, const double *D_mat, double *F_mat)
{
    int    nbf       = TinyDFT->nbf;
    int    mat_size  = TinyDFT->mat_size;
    int    *ipiv     = TinyDFT->DIIS_ipiv;
    double *F0_mat   = TinyDFT->F0_mat;
    double *R_mat    = TinyDFT->R_mat;
    double *B_mat    = TinyDFT->B_mat;
    double *FDS_mat  = TinyDFT->FDS_mat;
    double *DIIS_rhs = TinyDFT->DIIS_rhs;
    double *tmp_mat  = TinyDFT->tmp_mat;
    
    int mat_msize = DBL_SIZE * mat_size;
    int ldB = MAX_DIIS + 1;
    
    if (TinyDFT->iter <= 1)
    {
        // F = X^T * F * X
        // Use tmp_mat to store X^T * F
        cblas_dgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans, nbf, nbf, nbf, 
            1.0, X_mat, nbf, F_mat, nbf, 0.0, tmp_mat, nbf
        );
        // Use F_mat to store X^T * F * X
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf, 
            1.0, tmp_mat, nbf, X_mat, nbf, 0.0, F_mat, nbf
        );
        return;
    }
    
    int DIIS_idx;   // Which historic F matrix will be replaced
    if (TinyDFT->DIIS_len < MAX_DIIS)
    {
        DIIS_idx = TinyDFT->DIIS_len;
        TinyDFT->DIIS_len++;
    } else {
        DIIS_idx = TinyDFT->DIIS_bmax_id;
    }
    
    // FDS = F * D * S;
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf,
        1.0, F_mat, nbf, D_mat, nbf, 0.0, tmp_mat, nbf
    );
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf,
        1.0, tmp_mat, nbf, S_mat, nbf, 0.0, FDS_mat, nbf
    );
    
    // Residual = X^T * (FDS - FDS^T) * X, use tmp_mat to store FDS - FDS^T
    #pragma omp parallel for
    for (int i = 0; i < nbf; i++)
    {
        double *tmp_i = tmp_mat + i * nbf;
        double *FDS_mat_ri = FDS_mat + i * nbf;
        double *FDS_mat_ci = FDS_mat + i;
        #pragma omp simd
        for (int j = 0; j < nbf; j++)
            tmp_i[j] = FDS_mat_ri[j] - FDS_mat_ci[j * nbf];
    }
    
    // Use FDS_mat to store X^T * (FDS - FDS^T)
    cblas_dgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans, nbf, nbf, nbf,
        1.0, X_mat, nbf, tmp_mat, nbf, 0.0, FDS_mat, nbf
    );
    // Use tmp_mat to store X^T * (FDS - FDS^T) * X
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf,
        1.0, FDS_mat, nbf, X_mat, nbf, 0.0, tmp_mat, nbf
    );
    
    // In my MATLAB code, F_mat and its residual are treated as column vectors
    // For performance, we treat them as row vectors here
    
    // R(:, DIIS_idx) = X^T * (FDS - FDS^T) * X
    // B(i, j) = R(:, i) * R(:, j)
    // DIIS_rhs is not used yet, use it to store dot product results
    double *DIIS_dot = DIIS_rhs; 
    memset(DIIS_dot, 0, DBL_SIZE * (MAX_DIIS + 1));
    memcpy(R_mat + mat_size * DIIS_idx, tmp_mat, mat_msize);
    double *Ri = R_mat + mat_size * DIIS_idx;
    for (int j = 0; j < TinyDFT->DIIS_len; j++)
    {
        double *Rj = R_mat + mat_size * j;
        DIIS_dot[j] = cblas_ddot(mat_size, Ri, 1, Rj, 1);
    }
    
    // Construct symmetric B
    // B(DIIS_idx, 1 : DIIS_len) = DIIS_dot(1 : DIIS_idx);
    // B(1 : DIIS_len, DIIS_idx) = DIIS_dot(1 : DIIS_idx);
    for (int i = 0; i < TinyDFT->DIIS_len; i++)
    {
        B_mat[DIIS_idx * ldB + i] = DIIS_dot[i];
        B_mat[i * ldB + DIIS_idx] = DIIS_dot[i];
    }
    
    // Pick an old F that its residual has the largest 2-norm
    for (int i = 0; i < TinyDFT->DIIS_len; i++)
    {
        if (B_mat[i * ldB + i] > TinyDFT->DIIS_bmax)
        {
            TinyDFT->DIIS_bmax    = B_mat[i * ldB + i];
            TinyDFT->DIIS_bmax_id = i;
        }
    }
    
    // F := X^T * F * X, F0(:, DIIS_idx) = F
    // Use tmp_mat to store X^T * F
    cblas_dgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans, nbf, nbf, nbf, 
        1.0, X_mat, nbf, F_mat, nbf, 0.0, tmp_mat, nbf
    );
    // Use F_mat to store X^T * F * X
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf, 
        1.0, tmp_mat, nbf, X_mat, nbf, 0.0, F_mat, nbf
    );
    // Copy to F0
    memcpy(F0_mat + mat_size * DIIS_idx, F_mat, mat_msize);
    
    // Solve the linear system 
    memset(DIIS_rhs, 0, DBL_SIZE * (MAX_DIIS + 1));
    DIIS_rhs[TinyDFT->DIIS_len] = -1;
    // Copy B_mat to tmp_mat, since LAPACKE_dgesv will overwrite the input matrix
    memcpy(tmp_mat, B_mat, DBL_SIZE * ldB * ldB);  
    LAPACKE_dgesv(LAPACK_ROW_MAJOR, TinyDFT->DIIS_len + 1, 1, tmp_mat, ldB, ipiv, DIIS_rhs, 1);
    
    // Form new X^T * F * X
    memset(F_mat, 0, mat_msize);
    for (int i = 0; i < TinyDFT->DIIS_len; i++)
        cblas_daxpy(mat_size, DIIS_rhs[i], F0_mat + i * mat_size, 1, F_mat, 1);
}
