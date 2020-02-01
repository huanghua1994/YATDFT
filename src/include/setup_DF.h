#ifndef __SETUP_DF_H__
#define __SETUP_DF_H__

#include "TinyDFT_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Set up density fitting, including:
//   (1) load DF basis set and preparing ERI related data structures using libCMS;
//   (2) allocate memory for all tensors, matrices, and arrays used in DF;
//   (3) precompute some tensors and matrices used in DF.
// Input parameters:
//   TinyDFT      : Initialized TinyDFT structure
//   df_bas_fname : File name of Gaussian basis set for DF
// Output parameter:
//   TinyDFT : TinyDFT structure with initialized DF data structures
void TinyDFT_setup_DF(TinyDFT_t TinyDFT, char *df_bas_fname, char *xyz_fname);

#ifdef __cplusplus
}
#endif

#endif
