#ifndef __EVAL_XC_FUNC_H__
#define __EVAL_XC_FUNC_H__

// Functional IDs here are the same as Libxc
#define  LDA_X                     1  // Slater exchange
#define  LDA_C_XA                  6  // Slater Xalpha correlation
#define  LDA_C_PZ                  9  // Perdew & Zunger 81 correlation
#define  LDA_C_PW                 12  // Perdew & Wang 92 correlation

// Evaluate LDA XC functional E_xc = \int G(rho(r)) dr
// Input parameters:
//   fid : XC functional ID
//   np  : Total number of points
//   rho : Size np, electron density at each point
// Output parameters:
//   exc : Size np, "energy per unit particle", == G / rho
//   vxc : Size np, correlation potential, == \frac{\part G}{\part rho}
void eval_LDA_exc_vxc(const int fid, const int np, const double *rho, double *exc, double *vxc);

#endif
