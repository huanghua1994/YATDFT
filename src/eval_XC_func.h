#ifndef __EVAL_XC_FUNC_H__
#define __EVAL_XC_FUNC_H__

// Functional and family IDs here are the same as Libxc
#define  FAMILY_LDA                1
#define  FAMILY_GGA                2
#define  FAMILY_HYB_GGA           32
#define  LDA_X                     1  // Slater exchange
#define  LDA_C_XA                  6  // Slater Xalpha correlation
#define  LDA_C_PZ                  9  // Perdew & Zunger 81 correlation
#define  LDA_C_PW                 12  // Perdew & Wang 92 correlation
#define  GGA_X_PBE               101  // Perdew, Burke & Ernzerhof exchange
#define  GGA_X_B88               106  // Becke 88 exchange
#define  GGA_X_G96               107  // Gill 96 exchange
#define  GGA_X_PW86              108  // Perdew & Wang 86 exchange
#define  GGA_X_PW91              109  // Perdew & Wang 91 exchange
#define  GGA_C_PBE               130  // Perdew, Burke & Ernzerhof correlation
#define  GGA_C_LYP               131  // Lee, Yang & Parr correlation
#define  GGA_C_P86               132  // Perdew 86 correlation
#define  GGA_C_PW91              134  // Perdew & Wang 91 correlation
#define  HYB_GGA_XC_B3LYP        402  // The (in)famous B3LYP
#define  HYB_GGA_XC_B3LYP5       475  // B3LYP with VWN functional 5 instead of RPA

const static int num_impl_xc_func = 8;
const static int impl_xc_func[8]  = {
    LDA_X, LDA_C_XA, LDA_C_PZ, LDA_C_PW, 
    GGA_X_PBE, GGA_X_B88, 
    GGA_C_PBE, GGA_C_LYP//, GGA_C_P86  // GGA_C_P86 seems to have accuracy issue, fix it later
};

#ifdef __cplusplus
extern "C" {
#endif

// Evaluate LDA XC functional E_xc = \int G(rho(r)) dr
// Input parameters:
//   fid : XC functional ID
//   npt : Total number of points
//   rho : Size npt, electron density at each point
// Output parameters:
//   exc : Size npt, = G / rho
//   vxc : Size npt, = \frac{\part G}{\part rho}
void eval_LDA_exc_vxc(const int fid, const int npt, const double *rho, double *exc, double *vxc);

// Evaluate GGA XC functional E_xc = \int G(rho(r)) dr
// Input parameters:
//   fid   : XC functional ID
//   npt   : Total number of points
//   rho   : Size npt, electron density at each point
//   sigma : Size npt, contracted gradient of rho
// Output parameters:
//   exc    : Size npt, = G / rho
//   vrho   : Size npt, = \frac{\part G}{\part rho}
//   vsigma : Size npt, = \frac{\part G}{\part sigma}
void eval_GGA_exc_vxc(
    const int fid, const int npt, const double *rho, const double *sigma, 
    double *exc, double *vrho, double *vsigma
);

#ifdef __cplusplus
}
#endif

#endif
