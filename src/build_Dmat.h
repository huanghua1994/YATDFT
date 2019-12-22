#ifndef _YATSCF_BUILD_DMAT_H_
#define _YATSCF_BUILD_DMAT_H_

#include "TinySCF_typedef.h"

// All matrices used in this module is row-major, leading dimension = number of columns

#ifdef __cplusplus
extern "C" {
#endif

// Build initial density matrix using SAD (if SAD not available, D = 0)
// Input parameter:
//   TinySCF : Initialized TinySCF structure
// Output parameter:
//   D_mat : Initial density matrix, size nbf * nbf
void TinySCF_build_Dmat_SAD(TinySCF_t TinySCF, double *D_mat);

// Build density matrix using eigen-decomposition
// Input parameter:
//   TinySCF : Initialized TinySCF structure
//   F_mat   : Fock matrix after DIIS, size nbf * nbf
//   X_mat   : Basis transformation matrix, size nbf * nbf
// Output parameter:
//   D_mat    : Density matrix, size nbf * nbf, D = Cocc * Cocc^T
//   Cocc_mat : Cocc matrix, size nbf * n_occ
void TinySCF_build_Dmat_eig(TinySCF_t TinySCF, const double *F_mat, const double *X_mat, double *D_mat, double *Cocc_mat);

#ifdef __cplusplus
}
#endif

#endif
