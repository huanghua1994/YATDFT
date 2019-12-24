#ifndef __BUILD_XCMAT_H__
#define __BUILD_XCMAT_H__

#ifdef __cplusplus
extern "C" {
#endif

// Set up exchange-correlation numerical integral environments
// Input parameter:
//   TinyDFT : Initialized TinyDFT structure
// Output parameters:
//   TinyDFT : TinyDFT structure with XC integral environment ready
void TinyDFT_setup_XC_integral(TinyDFT_t TinyDFT);

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
