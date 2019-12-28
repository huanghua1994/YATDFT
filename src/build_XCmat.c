#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include <mkl.h>

#include "utils.h"
#include "TinyDFT_typedef.h"
#include "gen_int_grid.h"
#include "build_XCmat.h"
#include "eval_XC_func.h"

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

// Set up exchange-correlation numerical integral environments
void TinyDFT_setup_XC_integral(TinyDFT_t TinyDFT, const char *xf_str, const char *cf_str)
{
    TinyDFT_flatten_shell_info_to_bf(TinyDFT);
    
    gen_int_grid(
        TinyDFT->natom, TinyDFT->atom_xyz, TinyDFT->atom_idx,
        &TinyDFT->nintp, &TinyDFT->int_grid
    );
    
    int nbf   = TinyDFT->nbf;
    int nintp = TinyDFT->nintp;
    int nintp_blk = 1024;   // Calculate no more than 1024 points each time
    TinyDFT->nintp_blk = nintp_blk;
    TinyDFT->phi = (double*) ALIGN64B_MALLOC(DBL_SIZE * nintp_blk * nbf);
    TinyDFT->rho = (double*) ALIGN64B_MALLOC(DBL_SIZE * nintp_blk);
    TinyDFT->exc = (double*) ALIGN64B_MALLOC(DBL_SIZE * nintp_blk);
    TinyDFT->vxc = (double*) ALIGN64B_MALLOC(DBL_SIZE * nintp_blk);
    TinyDFT->XC_workbuf = (double*) ALIGN64B_MALLOC(DBL_SIZE * nintp_blk * (nbf + 4));
    assert(TinyDFT->phi != NULL);
    assert(TinyDFT->rho != NULL);
    assert(TinyDFT->exc != NULL);
    assert(TinyDFT->vxc != NULL);
    assert(TinyDFT->XC_workbuf != NULL);
    TinyDFT->mem_size += (double) (DBL_SIZE * nintp_blk * (2 * nbf + 7));
    
    TinyDFT->xfid = LDA_X;
    TinyDFT->cfid = LDA_C_XA;
    if (strcmp(cf_str, "LDA_C_XA") == 0) TinyDFT->cfid = LDA_C_XA;
    if (strcmp(cf_str, "LDA_C_PZ") == 0) TinyDFT->cfid = LDA_C_PZ;
    if (strcmp(cf_str, "LDA_C_PW") == 0) TinyDFT->cfid = LDA_C_PW;
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
static void TinySCF_eval_basis_func(
    const int nintp, const double *int_grid, const int sintp, const int eintp,
    const int nbf, const double *bf_coef, const double *bf_alpha, 
    const double *bf_exp, const double *bf_center, const int *bf_nprim,
    const int max_nprim,  const int ld_phi, double *phi
)
{
    const double *ipx = int_grid + 0 * nintp;
    const double *ipy = int_grid + 1 * nintp;
    const double *ipz = int_grid + 2 * nintp;
    
    #pragma omp parallel for
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
        double *phi_i = phi + i * ld_phi;
        
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
            phi_i[j  - sintp] = phi_ij;
        }
    }
}

// Evaluate electron density at given grid points
// Input parameters:
//   nbf     : Number of basis functions
//   D_mat   : Size nbf-by-nbf, density matrix
//   ld_phi  : Leading dimension of phi, == maximum number of grid point 
//             results per basis function that phi can store
//   npts    : The first npts phi values in phi will be used
//   phi     : Size nbf-by-ld_phi, phi[i, :] are i-th basis function values
//             at some integral points
//   workbuf : Size nbf-by-npts
// Output parameter:
//   rho : ELectron density corresponding to the integral points in the 
//         first npts grid points in phi
static void TinyDFT_eval_electron_density(
    const int nbf, const double *D_mat, 
    const int ld_phi, const int npts, const double *phi,
    double *workbuf, double *rho
)
{
    double *D_phi = workbuf;
    
    // rho_{g} = \sum_{u,v} phi_{u,g} * D_{u,v} * phi_{v,g} is the 
    // electron density at g-th grid point. 
    // (1) D_phi = D * phi;
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, npts, nbf, 
        1.0, D_mat, nbf, phi, ld_phi, 0.0, D_phi, npts
    );
    // (2) rho = 2 * sum_column(phi .* D_phi), "2 *" is that we use
    // D = Cocc*Cocc^T instead of D = 2*Cocc*Cocc^T outside
    int nthread = omp_get_num_threads();
    #pragma omp parallel num_threads(nthread)
    {
        int tid  = omp_get_thread_num();
        int spos = block_spos(nthread, tid,     npts);
        int epos = block_spos(nthread, tid + 1, npts);
        for (int j = spos; j < epos; j++) rho[j] = 0.0;
        for (int i = 0; i < nbf; i++)
        {
            const double *phi_i   = phi   + i * ld_phi;
            const double *D_phi_i = D_phi + i * npts;
            #pragma omp simd
            for (int j = spos; j < epos; j++)
                rho[j] += phi_i[j] * D_phi_i[j];
        }
        for (int j = spos; j < epos; j++) rho[j] *= 2.0;
    }
}

// Evaluate LDA XC functional and calculate XC energy
// Input parameters:
//   xfid    : Exchange functional ID
//   cfid    : Correlation functional ID
//   npts    : Number of points
//   rho     : Size npts, electron density at some integral points
//   ipw     : Size npts, numerical integral weights of points in rho
//   workbuf : Size npts * 4
// Output paramaters:
//   exc      : Size npts, "energy per unit particle", == G / rho
//   vxc      : Size npts, correlation potential, == \frac{\part G}{\part rho}
//   <return> : XC energy, E_xc = \int G(rho(r)) dr
static double TinyDFT_eval_LDA_XC_func(
    const int xfid, const int cfid, const int npts, const double *rho,
    const double *ipw, double *workbuf, double *exc, double *vxc
)
{
    double *ex = workbuf + npts * 0;
    double *ec = workbuf + npts * 1;
    double *vx = workbuf + npts * 2;
    double *vc = workbuf + npts * 3;

    eval_LDA_exc_vxc(xfid, npts, rho, ex, vx);
    eval_LDA_exc_vxc(cfid, npts, rho, ec, vc);
    
    double E_xc = 0.0;
    #pragma omp simd
    for (int i = 0; i < npts; i++)
    {
        exc[i] = ex[i] + ec[i];
        vxc[i] = vx[i] + vc[i];
        E_xc += exc[i] * rho[i] * ipw[i];
    }
    
    return E_xc;
}

// Build partial DFT XC matrix using LDA functional and rho
// and accumulate it to the final XC matrix
// Input parameters:
//   nbf     : Number of basis functions
//   ld_phi  : Leading dimension of phi, == maximum number of grid point 
//             results per basis function that phi can store
//   npts    : The first npts phi values in phi will be used
//   phi     : Size nbf-by-ld_phi, phi[i, :] are i-th basis function values
//             at some integral points
//   vxc     : Size npts, correlation potential, will be overwritten by vxc .* ipw
//   ipw     : Size npts, numerical integral weights of points in rho
//   beta    : 0.0 if this is the first call, otherwise 1.0
//   workbuf : Size npts * (nbf + 2)
// Output parameter:
//   XC_mat : Accumulated DFT XC matrix
static void TinyDFT_build_XC_LDA_partial(
    const int nbf, const int ld_phi, const int npts, 
    const double *phi, double *vxc, const double *ipw,
    const double beta, double *workbuf, double *XC_mat
)
{
    // XC_{u,v} = \int phi_u(r) * vxc(r) * phi_v(r) dr
    double *phi_vxc_w = workbuf + npts * 4;
    #pragma omp simd
    for (int i = 0; i < npts; i++) vxc[i] *= ipw[i];
    #pragma omp parallel for
    for (int i = 0; i < nbf; i++)
    {
        const double *phi_i = phi + i * ld_phi;
        double *phi_vxc_w_i = phi_vxc_w + i * npts;
        #pragma omp simd
        for (int j = 0; j < npts; j++)
            phi_vxc_w_i[j] = phi_i[j] * vxc[j];
    }
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, nbf, nbf, npts,
        1.0, phi, ld_phi, phi_vxc_w, npts, beta, XC_mat, nbf
    );
}

// Construct DFT exchange-correlation matrix
double TinyDFT_build_XC_mat(TinyDFT_t TinyDFT, const double *D_mat, double *XC_mat)
{ 
    int    nbf        = TinyDFT->nbf;
    int    xfid       = TinyDFT->xfid;
    int    cfid       = TinyDFT->cfid;
    int    nintp      = TinyDFT->nintp;
    int    nintp_blk  = TinyDFT->nintp_blk;
    int    max_nprim  = TinyDFT->max_nprim;
    int    *bf_nprim  = TinyDFT->bf_nprim;
    double *bf_coef   = TinyDFT->bf_coef;
    double *bf_alpha  = TinyDFT->bf_alpha;
    double *bf_exp    = TinyDFT->bf_exp;
    double *bf_center = TinyDFT->bf_center;
    double *phi       = TinyDFT->phi;
    double *rho       = TinyDFT->rho;
    double *exc       = TinyDFT->exc;
    double *vxc       = TinyDFT->vxc;
    double *int_grid  = TinyDFT->int_grid;
    double *ipw       = TinyDFT->int_grid + 3 * nintp;
    double *workbuf   = TinyDFT->XC_workbuf;

    double E_xc = 0.0;
    for (int sintp = 0; sintp < nintp; sintp += nintp_blk)
    {
        int eintp = sintp + nintp_blk;
        if (eintp > nintp) eintp = nintp;
        int npts = eintp - sintp;
        double beta = (sintp == 0) ? 0.0 : 1.0;
        double *curr_ipw = ipw + sintp;
        
        TinySCF_eval_basis_func(
            nintp, int_grid, sintp, eintp, 
            nbf, bf_coef, bf_alpha,
            bf_exp, bf_center, bf_nprim, 
            max_nprim, nintp_blk, phi
        );
        
        TinyDFT_eval_electron_density(
            nbf, D_mat, nintp_blk, npts, 
            phi, workbuf, rho
        );
        
        E_xc += TinyDFT_eval_LDA_XC_func(
            xfid, cfid, npts, rho, curr_ipw, 
            workbuf, exc, vxc
        );
        
        TinyDFT_build_XC_LDA_partial(
            nbf, nintp_blk, npts, phi, vxc, 
            curr_ipw, beta, workbuf, XC_mat
        );
    }
    
    return E_xc;
}

