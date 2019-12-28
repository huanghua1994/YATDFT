#ifndef __EVAL_XC_FUNC_H__
#define __EVAL_XC_FUNC_H__

// Functional and family IDs here are the same as Libxc
#define  FAMILY_LDA                1
#define  FAMILY_GGA                2
#define  LDA_X                     1  // Slater exchange
#define  LDA_C_XA                  6  // Slater Xalpha correlation
#define  LDA_C_PZ                  9  // Perdew & Zunger 81 correlation
#define  LDA_C_PW                 12  // Perdew & Wang 92 correlation

const static int num_impl_xc_func = 4;
const static int impl_xc_func[4]  = {LDA_X, LDA_C_XA, LDA_C_PZ, LDA_C_PW};

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
