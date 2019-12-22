#ifndef _YATSCF_BUILD_FOCK_H_
#define _YATSCF_BUILD_FOCK_H_

#include "TinySCF.h"

// Compute core Hamiltonian and overlap matrix, and generate basis transform matrix
void TinySCF_build_Hcore_S_X(TinySCF_t TinySCF, double *Hc, double *S, double *X);

void TinySCF_build_FockMat(TinySCF_t TinySCF);

#endif
