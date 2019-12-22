#ifndef _YATSCF_BUILD_HF_MAT_H_
#define _YATSCF_BUILD_HF_MAT_H_

#include "TinySCF_typedef.h"

// All matrices used in this module is row-major, size = nbf * nbf, leading dimension = nbf

// Construct core Hamiltonian, overlap, and basis transform matrices
// Input parameter:
//   TinySCF : Initialized TinySCF structure
// Output parameters:
//   Hcore_mat : Core Hamiltonian matrix
//   S_mat     : Overlap matrix
//   X_mat     : Basis transformation matrix
void TinySCF_build_Hcore_S_X_mat(TinySCF_t TinySCF, double *Hcore_mat, double *S_mat, double *X_mat);

// Construct Coulomb and HF exchange matrices
// Input parameters:
//   TinySCF : Initialized TinySCF structure
//   D_mat   : Density matrix
// Output parameters:
//   J_mat : Coulomb matrix
//   K_mat : HF exchange matrix
void TinySCF_build_JKmat(TinySCF_t TinySCF, const double *D_mat, double *J_mat, double *K_mat);

// Calculate Hartree-Fock energies
// Input parameters:
//   TinySCF   : Initialized TinySCF structure
//   D_mat     : Density matrix
//   Hcore_mat : Core Hamiltonian matrix
//   J_mat     : Coulomb matrix
//   K_mat     : HF exchange matrix, can be NULL
// Output parameters:
//   *E_one_elec    : One-electron energy, == sum(sum(2 .* D .* Hcore))
//   *E_two_elec    : Two-electron integral energy, == sum(sum(2 .* D .* J))
//   *E_HF_exchange : Hartree-Fock exchange energy, == sum(sum(-D .* K)), will be ignored if K_mat == 0
void TinySCF_calc_HF_energy(
    const int mat_size, const double *D_mat, const double *Hcore_mat, const double *J_mat, 
    const double *K_mat, double *E_one_elec, double *E_two_elec, double *E_HF_exchange
);

#endif
