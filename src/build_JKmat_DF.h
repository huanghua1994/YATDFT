#ifndef __BUILD_JKMAT_DF_H__
#define __BUILD_JKMAT_DF_H__

#include "TinyDFT_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// All matrices used in this module is row-major, size = nbf * nbf, leading dimension = nbf

// Construct Coulomb and HF exchange matrices using density fitting
// Input parameters:
//   TinyDFT  : Initialized TinyDFT structure
//   D_mat    : Density matrix
//   Cocc_mat : D := Cocc * Cocc^T
// Output parameters:
//   J_mat : Coulomb matrix, == NULL will skip the construction of J_mat
//   K_mat : HF exchange matrix, == NULL will skip the construction of K_mat
void TinyDFT_build_JKmat_DF(TinyDFT_t TinyDFT, const double *D_mat, const double *Cocc_mat, double *J_mat, double *K_mat);

#ifdef __cplusplus
}
#endif

#endif

