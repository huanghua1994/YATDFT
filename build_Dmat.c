#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <omp.h>

#include <mkl.h>

#include "utils.h"
#include "TinySCF_typedef.h"
#include "build_Dmat.h"

void TinySCF_build_Dmat_SAD(TinySCF_t TinySCF, double *D_mat)
{
    assert(TinySCF != NULL);
    
    int natom    = TinySCF->natom;
    int charge   = TinySCF->charge;
    int electron = TinySCF->electron;
    int nbf      = TinySCF->nbf;
    int mat_size = TinySCF->mat_size;
    BasisSet_t basis = TinySCF->basis;
    
    memset(D_mat, 0, DBL_SIZE * mat_size);
    
    double *guess;
    int spos, epos, ldg;
    for (int i = 0; i < natom; i++)
    {
        CMS_getInitialGuess(basis, i, &guess, &spos, &epos);
        ldg = epos - spos + 1;
        double *D_mat_ptr = D_mat + spos * nbf + spos;
        copy_dbl_mat_blk(D_mat_ptr, nbf, guess, ldg, ldg, ldg);
    }
    
    // Scaling the initial density matrix according to the charge and neutral
    double R = 1.0;
    if (charge != 0 && electron != 0) 
        R = (double)(electron - charge) / (double)(electron);
    R *= 0.5;
    for (int i = 0; i < mat_size; i++) D_mat[i] *= R;
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
            ev_idx[j] = ev_idx[i];
            
            dswap     = eigval[i];
            eigval[i] = eigval[j];
            eigval[j] = dswap;
            
            i++;  j--;
        }
    }
    if (i < r) quickSort(eigval, ev_idx, i, r);
    if (j > l) quickSort(eigval, ev_idx, l, j);
}

void TinySCF_build_Dmat_eig(TinySCF_t TinySCF, const double *F_mat, const double *X_mat, double *D_mat, double *Cocc_mat)
{
    int    nbf       = TinySCF->nbf;
    int    n_occ     = TinySCF->n_occ;
    int    mat_size  = TinySCF->mat_size;
    int    *ev_idx   = TinySCF->ev_idx;
    double *eigval   = TinySCF->eigval;
    double *tmp_mat  = TinySCF->tmp_mat;

    // Notice: here F_mat is already = X^T * F * X
    memcpy(tmp_mat, F_mat, DBL_SIZE * TinySCF->mat_size);
    
    // Diagonalize F = C0^T * epsilon * C0, and C = X * C0 
    // [C0, E] = eig(F1), now C0 is stored in tmp_mat
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', nbf, tmp_mat, nbf, eigval);  // tmp_mat will be overwritten by eigenvectors
    // C = X * C0, now C is stored in D_mat
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, nbf, nbf, 
        1.0, X_mat, nbf, tmp_mat, nbf, 0.0, D_mat, nbf
    );
    
    // Form the C_occ with eigenvectors corresponding to n_occ smallest eigenvalues
    for (int i = 0; i < nbf; i++) ev_idx[i] = i;
    quickSort(eigval, ev_idx, 0, nbf - 1);
    for (int j = 0; j < n_occ; j++)
        for (int i = 0; i < nbf; i++)
            Cocc_mat[i * n_occ + j] = D_mat[i * nbf + ev_idx[j]];
    
    // D = C_occ * C_occ^T
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, nbf, nbf, n_occ, 
        1.0, Cocc_mat, n_occ, Cocc_mat, n_occ, 0.0, D_mat, nbf
    );
}

