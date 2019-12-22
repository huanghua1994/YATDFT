#ifndef _YATSCF_DIIS_H_
#define _YATSCF_DIIS_H_

#include "TinySCF_typedef.h"

// DIIS acceleration (Pulay mixing)
void TinySCF_DIIS(TinySCF_t TinySCF, const double *X_mat, const double *S_mat, const double *D_mat, double *F_mat);

#endif
