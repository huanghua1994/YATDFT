#ifndef _YATSCF_BUILD_HF_MAT_H_
#define _YATSCF_BUILD_HF_MAT_H_

#include "TinySCF.h"

// Compute core Hamiltonian, overlap, and basis transform matrices
void TinySCF_build_Hcore_S_X_mat(TinySCF_t TinySCF, double *Hcore_mat, double *S_mat, double *X_mat);

// Compute Coulomb and HF exchange matrices
void TinySCF_build_JKmat(TinySCF_t TinySCF, const double *D_mat, double *J_mat, double *K_mat);

#endif
