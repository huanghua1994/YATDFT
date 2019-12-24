#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <mkl.h>

#include "utils.h"
#include "TinyDFT_typedef.h"
#include "gen_int_grid.h"
#include "build_XCmat.h"

// Flatten shell info to basis function info for XC calculation
// Input parameter:
//   TinyDFT : Initialized TinyDFT structure
// Output parameters:
//   TinyDFT : TinyDFT structure with Flattened basis function
static void TinyDFT_flatten_shell_info_to_bf(TinyDFT_t TinyDFT)
{
    int natom  = TinyDFT->natom;
    int nshell = TinyDFT->nshell;
    int nbf    = TinyDFT->nbf;
    BasisSet_t basis  = TinyDFT->basis;
    Simint_t   simint = TinyDFT->simint;
    
    // Allocate memory for flattened Gaussian basis function and atom info 
    // used only in XC calculation
    int max_nprim = 1;
    for (int i = 0; i < nshell; i++)
    {
        int nprim_i = basis->nexp[i];
        if (nprim_i > max_nprim) max_nprim = nprim_i;
    }
    TinyDFT->max_nprim = max_nprim;
    TinyDFT->atom_idx  = (int*)    ALIGN64B_MALLOC(INT_SIZE * natom);
    TinyDFT->bf_nprim  = (int*)    ALIGN64B_MALLOC(INT_SIZE * nbf);
    TinyDFT->atom_xyz  = (double*) ALIGN64B_MALLOC(DBL_SIZE * 3 * natom);
    TinyDFT->bf_coef   = (double*) ALIGN64B_MALLOC(DBL_SIZE * nbf * max_nprim);
    TinyDFT->bf_alpha  = (double*) ALIGN64B_MALLOC(DBL_SIZE * nbf * max_nprim);
    TinyDFT->bf_exp    = (double*) ALIGN64B_MALLOC(DBL_SIZE * nbf * 3);
    TinyDFT->bf_center = (double*) ALIGN64B_MALLOC(DBL_SIZE * nbf * 3);
    assert(TinyDFT->atom_idx  != NULL);
    assert(TinyDFT->bf_nprim  != NULL);
    assert(TinyDFT->atom_xyz  != NULL);
    assert(TinyDFT->bf_coef   != NULL);
    assert(TinyDFT->bf_alpha  != NULL);
    assert(TinyDFT->bf_exp    != NULL);
    assert(TinyDFT->bf_center != NULL);
    TinyDFT->mem_size += (double) (INT_SIZE * (natom + nbf));
    TinyDFT->mem_size += (double) (DBL_SIZE * 3 * natom);
    TinyDFT->mem_size += (double) (DBL_SIZE * nbf * 2 * (max_nprim + 3));
    memset(TinyDFT->bf_coef,  0, DBL_SIZE * nbf * max_nprim);
    memset(TinyDFT->bf_alpha, 0, DBL_SIZE * nbf * max_nprim);
    
    for (int iatom = 0; iatom < natom; iatom++)
    {
        TinyDFT->atom_xyz[0 * natom + iatom] = basis->xn[iatom];
        TinyDFT->atom_xyz[1 * natom + iatom] = basis->yn[iatom];
        TinyDFT->atom_xyz[2 * natom + iatom] = basis->zn[iatom];
        TinyDFT->atom_idx[iatom] = basis->eid[iatom];
    }
    
    int ibf = 0;
    for (int i = 0; i < nshell; i++)
    {
        int    am_i     = simint->shells[i].am;
        int    nprim_i  = simint->shells[i].nprim;
        int    nbf_i    = (am_i+2) * (am_i+1) / 2;
        double x_i      = simint->shells[i].x;
        double y_i      = simint->shells[i].y;
        double z_i      = simint->shells[i].z;
        double *coef_i  = simint->shells[i].coef;
        double *alpha_i = simint->shells[i].alpha;
        size_t cp_msize = DBL_SIZE * nprim_i;
        for (int j = ibf; j < ibf + nbf_i; j++)
        {
            memcpy(TinyDFT->bf_coef  + j * max_nprim, coef_i,  cp_msize);
            memcpy(TinyDFT->bf_alpha + j * max_nprim, alpha_i, cp_msize);
            TinyDFT->bf_center[3 * j + 0] = x_i;
            TinyDFT->bf_center[3 * j + 1] = y_i;
            TinyDFT->bf_center[3 * j + 2] = z_i;
            TinyDFT->bf_nprim[j] = nprim_i;
        }
        
        for (int xe = am_i; xe >= 0; xe--)
        {
            for (int ye = am_i - xe; ye >= 0; ye--)
            {
                int ze = am_i - xe - ye;
                TinyDFT->bf_exp[3 * ibf + 0] = xe;
                TinyDFT->bf_exp[3 * ibf + 1] = ye;
                TinyDFT->bf_exp[3 * ibf + 2] = ze;
                ibf++;
            }
        }
    }
}

// Evaluate basis functions at specified integral grid points
// Input parameters:
//   nintp     : Total number of XC numerical integral points
//   int_grid  : Size 4-by-nintp, integral points and weights
//   sintp     : First integral point to evaluate
//   eintp     : Last integral point to evaluate + 1
//   nbf       : Number of basis functions
//   bf_coef   : Size nbf-by-max_nprim, coef  terms of basis functions
//   bf_alpha  : Size nbf-by-max_nprim, alpha terms of basis functions
//   bf_exp    : Size nbf-by-3, polynomial exponents terms of basis functions
//   bf_center : Size nbf-by-3, center of basis functions
//   bf_nprim  : Size nbf, number of primitive functions in each basis function
//   max_nprim : Maximum number of primitive functions in all basis functions
//   ld_phi    : Leading dimension of phi, == maximum number of grid point 
//               results per basis function that phi can store
// Output parameter:
//   phi : Size nbf-by-ld_phi, phi[i, :] are i-th basis function values
//         at integral points [sintp, eintp)
static void TinySCF_eval_BF_at_int_grid(
    const int nintp, const double *int_grid, const int sintp, const int eintp,
    const int nbf, const double *bf_coef, const double *bf_alpha, 
    const double *bf_exp, const double *bf_center, const int *bf_nprim,
    const int max_nprim,  const int ld_phi, double *phi
)
{
    const double *ipx = int_grid + 0 * nintp;
    const double *ipy = int_grid + 1 * nintp;
    const double *ipz = int_grid + 2 * nintp;
    const double *ipw = int_grid + 3 * nintp;
    
    // TODO: OpenMP parallelize this loop
    for (int i = 0; i < nbf; i++)
    {
        const int    bfnp = bf_nprim[i];
        const double bfx  = bf_center[3 * i + 0];
        const double bfy  = bf_center[3 * i + 1];
        const double bfz  = bf_center[3 * i + 2];
        const double bfxe = bf_exp[3 * i + 0];
        const double bfye = bf_exp[3 * i + 1];
        const double bfze = bf_exp[3 * i + 2];
        const double *bfc = bf_coef  + i * max_nprim;
        const double *bfa = bf_alpha + i * max_nprim;
        
        #pragma omp simd
        for (int j = sintp; j < eintp; j++)
        {
            double dx = ipx[j] - bfx;
            double dy = ipy[j] - bfy;
            double dz = ipz[j] - bfz;
            double poly = pow(dx, bfxe) * pow(dy, bfye) * pow(dz, bfze);
            double d2 = dx * dx + dy * dy + dz * dz;
            double phi_ij = 0.0;
            for (int p = 0; p < bfnp; p++)
                phi_ij += bfc[p] * poly * exp(-bfa[p] * d2);
            phi[i * ld_phi + (j - sintp)] = phi_ij;
        }
    }
}

// Set up exchange-correlation numerical integral environments
void TinyDFT_setup_XC_integral(TinyDFT_t TinyDFT)
{
    TinyDFT_flatten_shell_info_to_bf(TinyDFT);
    
    gen_int_grid(
        TinyDFT->natom, TinyDFT->atom_xyz, TinyDFT->atom_idx,
        &TinyDFT->nintp, &TinyDFT->int_grid
    );
    
    int nbf   = TinyDFT->nbf;
    int nintp = TinyDFT->nintp;
    TinyDFT->phi        = (double*) ALIGN64B_MALLOC(DBL_SIZE * nintp * nbf);
    TinyDFT->rho        = (double*) ALIGN64B_MALLOC(DBL_SIZE * nintp);
    TinyDFT->XC_workbuf = (double*) ALIGN64B_MALLOC(DBL_SIZE * nintp * (nbf + 2));
    assert(TinyDFT->phi        != NULL);
    assert(TinyDFT->rho        != NULL);
    assert(TinyDFT->XC_workbuf != NULL);
    TinyDFT->mem_size += (double) (DBL_SIZE * nintp * (2 * nbf + 3));
}

// Evaluate electron density at given grid points
// Input parameters:
//   nbf     : Number of basis functions
//   D_mat   : Size nbf-by-nbf, density matrix
//   ld_phi  : Leading dimension of phi, == maximum number of grid point 
//             results per basis function that phi can store
//   npt_phi : The first npt_phi phi values in phi will be used
//   phi     : Size nbf-by-ld_phi, phi[i, :] are i-th basis function values
//             at some integral points
//   workbuf : Size nbf-by-npt_phi
// Output parameter:
//   rho : ELectron density corresponding to the integral points in the 
//         first npt_phi grid points in phi
static void TinyDFT_eval_electron_density(
    const int nbf, const double *D_mat, 
    const int ld_phi, const int npt_phi, const double *phi,
    double *workbuf, double *rho
)
{
    double *D_phi = workbuf;
    
    // rho_{g} = \sum_{u,v} phi_{u,g} * D_{u,v} * phi_{v,g} is the 
    // electron density at g-th grid point. 
    // (1) D_phi = D * phi;
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, npt_phi, nbf, 
        1.0, D_mat, nbf, phi, ld_phi, 0.0, D_phi, npt_phi
    );
    // (2) rho = 2 * sum_column(phi .* D_phi), "2 *" is that we use
    // D = Cocc*Cocc^T instead of D = 2*Cocc*Cocc^T outside
    memset(rho, 0, DBL_SIZE * npt_phi);
    for (int i = 0; i < nbf; i++)
    {
        const double *phi_i   = phi   + i * ld_phi;
        const double *D_phi_i = D_phi + i * npt_phi;
        #pragma omp simd
        for (int j = 0; j < npt_phi; j++)
            rho[j] += phi_i[j] * D_phi_i[j];
    }
    for (int j = 0; j < npt_phi; j++) rho[j] *= 2.0;
}

// Build partial DFT XC matrix using Xalpha functional and rho
// and accumulate it to the final XC matrix
// Input parameters:
//   nbf     : Number of basis functions
//   ld_phi  : Leading dimension of phi, == maximum number of grid point 
//             results per basis function that phi can store
//   npt_phi : The first npt_phi phi values in phi will be used
//   phi     : Size nbf-by-ld_phi, phi[i, :] are i-th basis function values
//             at some integral points
//   rho     : Size npt_phi, electron density at some integral points
//   ipw     : Size npt_phi, numerical integral weights of points in rho
//   beta    : 0.0 if this is the first call, otherwise 1.0
//   workbuf : Size npt_phi * (nbf + 2)
// Output parameter:
//   XC_mat : Accumulated DFT XC matrix
//   E_xc   : Sum of XC energy on the given grid points
static double TinyDFT_build_XC_Xalpha_partial(
    const int nbf, const int ld_phi, const int npt_phi, 
    const double *phi, const double *rho, const double *ipw,
    const double beta, double *workbuf, double *XC_mat
)
{
    // Xalpha energy: Exc = \int -alpha * (9/8) * (3/pi)^(1/3) * rho(r)^(4/3) dr
    // Xalpha potential: vxc(r) = \frac{\delta Exc}{\delta rho}
    // Exc = \int exc(r) * rho(r) dr
    // XC_{u,v} = \int phi_u(r) * vxc(r) * phi_v(r) dr
    // We use exc and vxc here for using Libxc in the future
    double E_xc = 0.0;
    double *exc = workbuf;
    double *vxc = exc + npt_phi;
    double *phi_vxc_w = vxc + npt_phi;
    
    // (1) Evaluate exc, vxc, and XC energy
    double alpha = 0.7;
    double vxc_coef = -alpha * (3.0/2.0) * pow(3.0/M_PI, 1.0/3.0);
    double exc_coef = -alpha * (9.0/8.0) * pow(3.0/M_PI, 1.0/3.0);
    #pragma omp simd
    for (int i = 0; i < npt_phi; i++) 
    {
        double rho_i_13 = pow(rho[i], 1.0/3.0);
        exc[i] = exc_coef * rho_i_13;
        vxc[i] = vxc_coef * rho_i_13;
        E_xc += exc[i] * rho[i] * ipw[i];
    }
    
    // (2) XC_{u,v} = \int phi_u(r) * vxc(r) * phi_v(r) dr
    for (int i = 0; i < npt_phi; i++) vxc[i] *= ipw[i];
    for (int i = 0; i < nbf; i++)
    {
        const double *phi_i = phi + i * ld_phi;
        double *phi_vxc_w_i = phi_vxc_w + i * npt_phi;
        for (int j = 0; j < npt_phi; j++)
            phi_vxc_w_i[j] = phi_i[j] * vxc[j];
    }
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, nbf, nbf, npt_phi,
        1.0, phi, ld_phi, phi_vxc_w, npt_phi, beta, XC_mat, nbf
    );
    
    return E_xc;
}

double TinyDFT_build_XC_mat(TinyDFT_t TinyDFT, const double *D_mat, double *XC_mat)
{
    double E_xc = 0.0;
    
    int nbf   = TinyDFT->nbf;
    int nintp = TinyDFT->nintp;
    double *phi = TinyDFT->phi;
    double *rho = TinyDFT->rho;
    double *ipw = TinyDFT->int_grid + 3 * nintp;
    double *workbuf = TinyDFT->XC_workbuf;
    
    // TODO: slice nintp into multiple segments, evaluate BF values 
    // segment by segment and accumulate the XC matrix

    TinySCF_eval_BF_at_int_grid(
        nintp, TinyDFT->int_grid, 0, nintp, 
        nbf, TinyDFT->bf_coef, TinyDFT->bf_alpha,
        TinyDFT->bf_exp, TinyDFT->bf_center, TinyDFT->bf_nprim, 
        TinyDFT->max_nprim, nintp, TinyDFT->phi
    );

    TinyDFT_eval_electron_density(nbf, D_mat, nintp, nintp, phi, workbuf, rho);
    
    E_xc = TinyDFT_build_XC_Xalpha_partial(
        nbf, nintp, nintp, phi, rho, 
        ipw, 0.0, workbuf, XC_mat
    );
    
    return E_xc;
}

