#ifndef _YATSCF_BUILD_DMAT_H_
#define _YATSCF_BUILD_DMAT_H_

#include "TinySCF.h"

// Build density matrix using eigen decomposition (diagonalization)
void TinySCF_build_Dmat_eig(TinySCF_t TinySCF, const double *F_mat, const double *X_mat, double *D_mat, double *Cocc_mat);

#endif
