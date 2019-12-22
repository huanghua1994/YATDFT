#ifndef _YATDFT_CDIIS_H_
#define _YATDFT_CDIIS_H_

#include "TinyDFT_typedef.h"

#define MAX_DIIS 10
#define MIN_DIIS 2

// All matrices used in this module is row-major, size = nbf * nbf, leading dimension = nbf

#ifdef __cplusplus
extern "C" {
#endif

// CDIIS acceleration (Pulay mixing)
// Input parameters:
//   TinyDFT : Initialized TinyDFT structure
//   X_mat   : Basis transformation matrix
//   S_mat   : Overlap matrix
//   D_mat   : Density matrix
//   F_mat   : Fock matrix
// Output parameter:
//   F_mat : CDIIS processed X^T * F * X 
void TinyDFT_CDIIS(TinyDFT_t TinyDFT, const double *X_mat, const double *S_mat, const double *D_mat, double *F_mat);

#ifdef __cplusplus
}
#endif

#endif
