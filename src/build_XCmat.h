#ifndef __BUILD_XCMAT_H__
#define __BUILD_XCMAT_H__

#ifdef __cplusplus
extern "C" {
#endif

// Set up exchange-correlation numerical integral environments
// Input parameter:
//   TinyDFT : Initialized TinyDFT structure
//   xf_str  : Exchange function string, default is "LDA_X" (Slater exchange)
//   cf_str  : Correlation function string, default is "LDA_C_XALPHA" (Slater's Xalpha)
// Output parameters:
//   TinyDFT : TinyDFT structure with XC integral environment ready
void TinyDFT_setup_XC_integral(TinyDFT_t TinyDFT, const char *xf_str, const char *cf_str);

// Construct DFT exchange-correlation matrix
// Input parameters:
//   TinyDFT : Initialized TinyDFT structure
//   D_mat   : Density matrix
// Output parameters:
//   XC_mat   : DFT exchange-correlation matrix
//   <return> : DFT exchange-correlation energy
double TinyDFT_build_XC_mat(TinyDFT_t TinyDFT, const double *D_mat, double *XC_mat);

#ifdef __cplusplus
}
#endif

#endif
