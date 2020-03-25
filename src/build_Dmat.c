#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <omp.h>

#include "linalg_lib_wrapper.h"

#include "utils.h"
#include "TinyDFT_typedef.h"
#include "build_Dmat.h"

void TinyDFT_build_Dmat_SAD(TinyDFT_t TinyDFT, double *D_mat)
{
    assert(TinyDFT != NULL);
    
    int natom    = TinyDFT->natom;
    int charge   = TinyDFT->charge;
    int electron = TinyDFT->electron;
    int nbf      = TinyDFT->nbf;
    int mat_size = TinyDFT->mat_size;
    BasisSet_t basis = TinyDFT->basis;
    
    memset(D_mat, 0, DBL_MSIZE * mat_size);
    
    double *guess;
    int spos, epos, ldg;
    for (int i = 0; i < natom; i++)
    {
        CMS_getInitialGuess(basis, i, &guess, &spos, &epos);
        ldg = epos - spos + 1;
        double *D_mat_ptr = D_mat + spos * nbf + spos;
        copy_dbl_mat_blk(guess, ldg, ldg, ldg, D_mat_ptr, nbf);
    }
    
    // Scaling the initial density matrix according to the charge and neutral
    double R = 1.0;
    if (charge != 0 && electron != 0) 
        R = (double)(electron - charge) / (double)(electron);
    R *= 0.5;
    for (int i = 0; i < mat_size; i++) D_mat[i] *= R;
}

void TinyDFT_build_Cocc_from_Dmat(TinyDFT_t TinyDFT, const double *D_mat, double *Cocc_mat)
{
    int    nbf       = TinyDFT->nbf;
    int    n_occ     = TinyDFT->n_occ;
    int    mat_size  = TinyDFT->mat_size;
    double *Chol_mat = TinyDFT->tmp_mat;
    
    int rank;
    int *piv = (int*) malloc(sizeof(int) * nbf);
    memcpy(Chol_mat, D_mat, DBL_MSIZE * mat_size);
    LAPACKE_dpstrf(LAPACK_ROW_MAJOR, 'L', nbf, Chol_mat, nbf, piv, &rank, 1e-12);
    
    if (rank < n_occ)
    {
        for (int i = 0; i < nbf; i++)
        {
            double *Chol_row = Chol_mat + i * nbf;
            for (int j = rank; j < n_occ; j++) Chol_row[j] = 0.0;
        }
    }
    
    for (int i = 0; i < n_occ; i++)
    {
        double *Cocc_row = Cocc_mat + i * n_occ;
        double *Chol_row = Chol_mat + i * nbf;
        for (int j = 0; j < i; j++) Cocc_row[j] = Chol_row[j];
        for (int j = i; j < n_occ; j++) Cocc_row[j] = 0.0;
    }
    for (int i = n_occ; i < nbf; i++)
    {
        double *Cocc_row = Cocc_mat + i * n_occ;
        double *Chol_row = Chol_mat + i * nbf;
        memcpy(Cocc_row, Chol_row, DBL_MSIZE * n_occ);
    }
    
    free(piv);
}

static void quickSort(double *eigval, int *ev_idx, int l, int r)
{
    int i = l, j = r, iswap;
    double mid = eigval[(i + j) / 2], dswap;
    while (i <= j)
    {
        while (eigval[i] < mid) i++;
        while (eigval[j] > mid) j--;
        if (i <= j)
        {
            iswap     = ev_idx[i];
            ev_idx[i] = ev_idx[j];
            ev_idx[j] = iswap;
            
            dswap     = eigval[i];
            eigval[i] = eigval[j];
            eigval[j] = dswap;
            
            i++;  j--;
        }
    }
    if (i < r) quickSort(eigval, ev_idx, i, r);
    if (j > l) quickSort(eigval, ev_idx, l, j);
}

void TinyDFT_build_Dmat_eig(TinyDFT_t TinyDFT, const double *F_mat, const double *X_mat, double *D_mat, double *Cocc_mat)
{
    int    nbf       = TinyDFT->nbf;
    int    n_occ     = TinyDFT->n_occ;
    int    mat_size  = TinyDFT->mat_size;
    int    *ev_idx   = TinyDFT->ev_idx;
    double *eigval   = TinyDFT->eigval;
    double *tmp_mat  = TinyDFT->tmp_mat;
    double *C_mat    = TinyDFT->C_mat;

    // Notice: here F_mat is already = X^T * F * X
    memcpy(tmp_mat, F_mat, DBL_MSIZE * mat_size);
    
    // Diagonalize F = C^T * epsilon * C
    // [C, E] = eig(F1), now C is stored in tmp_mat, each column is a eigenvector
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', nbf, tmp_mat, nbf, eigval);  
    
    // Sort the eigenvectors according to the eigenvalues
    for (int i = 0; i < nbf; i++) ev_idx[i] = i;
    quickSort(eigval, ev_idx, 0, nbf - 1);
    #pragma omp parallel for
    for (int j = 0; j < nbf; j++)
    {
        #pragma omp simd
        for (int i = 0; i < nbf; i++)
            C_mat[i * nbf + j] = tmp_mat[i * nbf + ev_idx[j]];
    }
    
    // C_occ = X * C(:, 1:n_occ)
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, n_occ, nbf, 
        1.0, X_mat, nbf, C_mat, nbf, 0.0, Cocc_mat, n_occ
    );
    
    // D = C_occ * C_occ^T
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, nbf, nbf, n_occ, 
        1.0, Cocc_mat, n_occ, Cocc_mat, n_occ, 0.0, D_mat, nbf
    );
}

void TinyDFT_build_Dmat_PD(TinyDFT_t TinyDFT, const double *F_mat, const double *X_mat, double *D_mat, double *Cocc_mat)
{
    int nbf   = TinyDFT->nbf;
    int n_occ = TinyDFT->n_occ;
    int n_vir = nbf - n_occ;
    int mat_size = TinyDFT->mat_size;
    
    double *eigval    = TinyDFT->eigval;
    double *C_mat     = TinyDFT->C_mat;
    double *tmp_mat   = TinyDFT->tmp_mat;
    double *Rot_mat   = tmp_mat + mat_size;
    double *C_prev    = Rot_mat + mat_size;
    double *workbuf   = C_prev  + mat_size;
    
    // Calculate the rotation angles X(i, a) and the threshold
    double *Cocc_prev = C_mat;
    double *Cvir_prev = C_mat + n_occ;
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, n_vir, nbf,
        1.0, F_mat, nbf, Cvir_prev, nbf, 0.0, tmp_mat, n_vir
    );
    cblas_dgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans, n_occ, n_vir, nbf,
        1.0, Cocc_prev, nbf, tmp_mat, n_vir, 0.0, Rot_mat, n_vir
    );
    double *Eocc_prev = eigval;
    double *Evir_prev = eigval + n_occ;
    #pragma omp parallel for
    for (int i = 0; i < n_occ; i++)
    {
        #pragma omp simd
        for (int a = 0; a < n_vir; a++)
            Rot_mat[i * n_vir + a] /= (Eocc_prev[i] - Evir_prev[a]);
    }
    double Rot_tol = 0.0;
    for (int i = 0; i < n_occ * n_vir; i++)
        if (fabs(Rot_mat[i]) > Rot_tol) Rot_tol = fabs(Rot_mat[i]);
    Rot_tol *= 0.04;
    
    // Transpose previous C matrix for better performance below
    #pragma omp parallel for
    for (int i = 0; i < nbf; i++)
    {
        #pragma omp simd
        for (int j = 0; j < nbf; j++)
        {
            C_prev[i * nbf + j]  = C_mat[j * nbf + i];
            workbuf[i * nbf + j] = C_mat[j * nbf + i];
        }
    }
    Cocc_prev = C_prev;
    Cvir_prev = C_prev + n_occ * nbf;
    
    // Perform Givens rotation between occupied and virtual eigenvectors
    double *Cocc_new = workbuf;
    #pragma omp parallel for
    for (int i = 0; i < n_occ; i++)
    {
        double *Cocc_new_i = Cocc_new + i * nbf;
        for (int a = 0; a < n_vir; a++)
        {
            if (fabs(Rot_mat[i * n_vir + a]) < Rot_tol) continue;
            double *Cvir_a = Cvir_prev + a * nbf;
            double s = Rot_mat[i * n_vir + a];
            double c = sqrt(1.0 - s * s);
            #pragma omp simd
            for (int k = 0; k < nbf; k++)
                Cocc_new_i[k] = c * Cocc_new_i[k] - s * Cvir_a[k];
        }
    }
    
    // C_occ = X * C(:, 1:n_occ)
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, nbf, n_occ, nbf, 
        1.0, X_mat, nbf, Cocc_new, nbf, 0.0, Cocc_mat, n_occ
    );
    
    // D = C_occ * C_occ^T
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, nbf, nbf, n_occ, 
        1.0, Cocc_mat, n_occ, Cocc_mat, n_occ, 0.0, D_mat, nbf
    );
}
