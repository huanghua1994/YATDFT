#ifndef _YATSCF_BUILD_DMAT_H_
#define _YATSCF_BUILD_DMAT_H_

#include "TinySCF_typedef.h"

// Build initial density matrix using SAD (if SAD not available, D = 0)
void TinySCF_build_Dmat_SAD(TinySCF_t TinySCF, double *D_mat);

// Build density matrix using eigen decomposition (diagonalization)
void TinySCF_build_Dmat_eig(TinySCF_t TinySCF, const double *F_mat, const double *X_mat, double *D_mat, double *Cocc_mat);

#endif
