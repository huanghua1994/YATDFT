#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include <mkl.h>
#include <xc.h>

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
    TinyDFT->phi        = (double*) ALIGN64B_MALLOC(DBL_SIZE * nintp_blk * nbf * 4);
    TinyDFT->rho        = (double*) ALIGN64B_MALLOC(DBL_SIZE * nintp_blk * 5);
    TinyDFT->exc        = (double*) ALIGN64B_MALLOC(DBL_SIZE * nintp_blk);
    TinyDFT->vxc        = (double*) ALIGN64B_MALLOC(DBL_SIZE * nintp_blk);
    TinyDFT->vsigma     = (double*) ALIGN64B_MALLOC(DBL_SIZE * nintp_blk);
    TinyDFT->XC_workbuf = (double*) ALIGN64B_MALLOC(DBL_SIZE * nintp_blk * (nbf + 6));
    assert(TinyDFT->phi        != NULL);
    assert(TinyDFT->rho        != NULL);
    assert(TinyDFT->exc        != NULL);
    assert(TinyDFT->vxc        != NULL);
    assert(TinyDFT->vsigma     != NULL);
    assert(TinyDFT->XC_workbuf != NULL);
    TinyDFT->mem_size += (double) (DBL_SIZE * nintp_blk * nbf * 4);
    TinyDFT->mem_size += (double) (DBL_SIZE * nintp_blk * 8);
    TinyDFT->mem_size += (double) (DBL_SIZE * nintp_blk * (nbf + 6));
    
    TinyDFT->xf_id = LDA_X;
    TinyDFT->cf_id = LDA_C_XA;
    TinyDFT->xf_family = FAMILY_LDA;
    TinyDFT->cf_family = FAMILY_LDA;
    
    if (strcmp(xf_str, "LDA_X")      == 0) { TinyDFT->xf_id = LDA_X;      TinyDFT->xf_family = FAMILY_LDA; }
    if (strcmp(cf_str, "LDA_C_XA")   == 0) { TinyDFT->cf_id = LDA_C_XA;   TinyDFT->cf_family = FAMILY_LDA; }
    if (strcmp(cf_str, "LDA_C_PZ")   == 0) { TinyDFT->cf_id = LDA_C_PZ;   TinyDFT->cf_family = FAMILY_LDA; }
    if (strcmp(cf_str, "LDA_C_PW")   == 0) { TinyDFT->cf_id = LDA_C_PW;   TinyDFT->cf_family = FAMILY_LDA; }
    
    if (strcmp(xf_str, "GGA_X_PBE")  == 0) { TinyDFT->xf_id = GGA_X_PBE;  TinyDFT->xf_family = FAMILY_GGA; }
    if (strcmp(xf_str, "GGA_X_B88")  == 0) { TinyDFT->xf_id = GGA_X_B88;  TinyDFT->xf_family = FAMILY_GGA; }
    if (strcmp(cf_str, "GGA_C_PBE")  == 0) { TinyDFT->cf_id = GGA_C_PBE;  TinyDFT->cf_family = FAMILY_GGA; }
    if (strcmp(cf_str, "GGA_C_P86")  == 0) { TinyDFT->cf_id = GGA_C_P86;  TinyDFT->cf_family = FAMILY_GGA; }
    
    TinyDFT->xf_impl = 0;
    TinyDFT->cf_impl = 0;
    for (int i = 0; i < num_impl_xc_func; i++)
    {
        if (TinyDFT->xf_id == impl_xc_func[i]) TinyDFT->xf_impl = 1;
        if (TinyDFT->cf_id == impl_xc_func[i]) TinyDFT->cf_impl = 1;
    }
    if (TinyDFT->xf_impl == 0)
    {
        int ret = xc_func_init(&TinyDFT->libxc_xf, TinyDFT->xf_id, XC_UNPOLARIZED);
        if (ret != 0) 
        {
            printf("FATAL: initialize exchange functional %d failed!\n", TinyDFT->xf_id);
            assert(ret == 0);
        }
        TinyDFT->xf_family = TinyDFT->libxc_xf.info->family;
    }
    if (TinyDFT->cf_impl == 0)
    {
        int ret = xc_func_init(&TinyDFT->libxc_cf, TinyDFT->cf_id, XC_UNPOLARIZED);
        if (ret != 0) 
        {
            printf("FATAL: initialize exchange functional %d failed!\n", TinyDFT->cf_id);
            assert(ret == 0);
        }
        TinyDFT->cf_family = TinyDFT->libxc_cf.info->family;
    }
    if (TinyDFT->xf_family != TinyDFT->cf_family)
    {
        printf("FATAL: exchange and correlation functionals are not the same family!\n");
        assert(TinyDFT->xf_family == TinyDFT->cf_family);
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
//   phi : Size 4*nbf-by-ld_phi. In each nbf-by-ld_phi block,
//         the i-th row is the i-th basis function values (its 1st order 
//         derivatives on x, y, z directions) at some integral points
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
    
    double *dphi_dx = phi + nbf * ld_phi * 1;
    double *dphi_dy = phi + nbf * ld_phi * 2;
    double *dphi_dz = phi + nbf * ld_phi * 3;
    
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
        
        int offset_i = i * ld_phi;
        double *phi_i     = phi     + offset_i;
        double *dphi_dx_i = dphi_dx + offset_i;
        double *dphi_dy_i = dphi_dy + offset_i;
        double *dphi_dz_i = dphi_dz + offset_i;
        
        #pragma omp simd
        for (int j = sintp; j < eintp; j++)
        {
            double dx   = ipx[j] - bfx;
            double dy   = ipy[j] - bfy;
            double dz   = ipz[j] - bfz;
            double poly = pow(dx, bfxe) * pow(dy, bfye) * pow(dz, bfze);
            double d2   = dx * dx + dy * dy + dz * dz;
            double rdx  = (dx == 0.0) ? 0.0 : 1.0 / dx;
            double rdy  = (dy == 0.0) ? 0.0 : 1.0 / dy;
            double rdz  = (dz == 0.0) ? 0.0 : 1.0 / dz;
            double phi_ij_p, phi_ij = 0.0;
            double dphi_dx_ij = 0.0, dphi_dy_ij = 0.0, dphi_dz_ij = 0.0;
            for (int p = 0; p < bfnp; p++)
            {
                phi_ij_p    = bfc[p] * poly * exp(-bfa[p] * d2);
                phi_ij     += phi_ij_p;
                dphi_dx_ij += (bfxe * rdx - 2.0 * bfa[p] * dx) * phi_ij_p;
                dphi_dy_ij += (bfye * rdy - 2.0 * bfa[p] * dy) * phi_ij_p;
                dphi_dz_ij += (bfze * rdz - 2.0 * bfa[p] * dz) * phi_ij_p;
            }
            int j_idx = j - sintp;
            phi_i[j_idx]     = phi_ij;
            dphi_dx_i[j_idx] = dphi_dx_ij;
            dphi_dy_i[j_idx] = dphi_dy_ij;
            dphi_dz_i[j_idx] = dphi_dz_ij;
        }
    }
}

// Evaluate electron density at given grid points
// Input parameters:
//   nbf     : Number of basis functions
//   D_mat   : Size nbf-by-nbf, density matrix
//   npt     : The first npt phi values in phi will be used
//   ld_phi  : Leading dimension of phi, == maximum number of grid point 
//             results per basis function that phi can store
//   phi     : Size 4*nbf-by-ld_phi. In each nbf-by-ld_phi block,
//             the i-th row is the i-th basis function values (its 1st order 
//             derivatives on x, y, z directions) at some integral points
//   ld_rho  : Leading dimension of rho (should == ld_phi)
//   workbuf : Size nbf-by-npt
// Output parameter:
//   rho : ELectron density corresponding to the integral points in the 
//         first npt grid points in phi
static void TinyDFT_eval_electron_density(
    const int nbf, const double *D_mat, const int npt, const int ld_phi, 
    const double *phi, const int ld_rho, double *rho, double *workbuf
)
{
    double *D_phi = workbuf;
    
    const double *dphi_dx = phi + nbf * ld_phi * 1;
    const double *dphi_dy = phi + nbf * ld_phi * 2;
    const double *dphi_dz = phi + nbf * ld_phi * 3;
    
    double *drho_dx = rho + ld_rho * 1;
    double *drho_dy = rho + ld_rho * 2;
    double *drho_dz = rho + ld_rho * 3;
    double *sigma   = rho + ld_rho * 4;
    
    // At the g-th grid point:
    // (1) electron density: rho_{g} = \sum_{u,v} phi_{u,g} * D_{u,v} * phi_{v,g}
    // (2) 1st derivatives of rho_{g}: drho_dx{g}, drho_dy{g}, drho_dz{g}, where 
    //     drho_dk{g} = 2 \sum_{u,v} D_{u,v} * phi_{v,g} * dphi_dk_{u,g}
    // (3) contracted gradient of rho_{g}: sigma_{g} = drho_dx{g}^2 + drho_dz{g}^2 + drho_dz{g}^2
    
    // 1. D_phi_{u,g} = \sum_{v} D_{u,v} * phi_{v,g}
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, npt, nbf, 
        1.0, D_mat, nbf, phi, ld_phi, 0.0, D_phi, npt
    );
    
    // 2. rho_{g}    = 2 * \sum_{u} D_phi_{u,g} * phi_{u,g}
    //    drho_dk{g} = 2 * (2 * \sum_{u} D_phi_{u,g} * dphi_dk_{u,g})
    // Note: the "2 *" before \sum is that we use D = Cocc * Cocc^T
    //       instead of D = 2 * Cocc * Cocc^T outside
    int nthread = omp_get_num_threads();
    #pragma omp parallel num_threads(nthread)
    {
        int tid  = omp_get_thread_num();
        int spos = block_spos(nthread, tid,     npt);
        int epos = block_spos(nthread, tid + 1, npt);
        #pragma omp simd
        for (int g = spos; g < epos; g++) 
        {
            rho[g]     = 0.0;
            drho_dx[g] = 0.0;
            drho_dy[g] = 0.0;
            drho_dz[g] = 0.0;
        }
        for (int u = 0; u < nbf; u++)
        {
            int phi_offset = u * ld_phi;
            const double *phi_u     = phi     + phi_offset;
            const double *dphi_dx_u = dphi_dx + phi_offset;
            const double *dphi_dy_u = dphi_dy + phi_offset;
            const double *dphi_dz_u = dphi_dz + phi_offset;
            const double *D_phi_u   = D_phi   + u * npt;
            #pragma omp simd
            for (int g = spos; g < epos; g++)
            {
                double D_phi_ug = D_phi_u[g];
                rho[g]     += phi_u[g]     * D_phi_ug;
                drho_dx[g] += dphi_dx_u[g] * D_phi_ug;
                drho_dy[g] += dphi_dy_u[g] * D_phi_ug;
                drho_dz[g] += dphi_dz_u[g] * D_phi_ug;
            }
        }
        #pragma omp simd
        for (int g = spos; g < epos; g++) 
        {
            rho[g]     *= 2.0;
            drho_dx[g] *= 4.0;
            drho_dy[g] *= 4.0;
            drho_dz[g] *= 4.0;
            sigma[g]    = drho_dx[g] * drho_dx[g];
            sigma[g]   += drho_dy[g] * drho_dy[g];
            sigma[g]   += drho_dz[g] * drho_dz[g];
        }
    }
}

// Evaluate LDA XC functional and calculate XC energy
// Input parameters:
//   xf_id      : Exchange functional ID
//   cf_id      : Correlation functional ID
//   xf_impl    : If we has built-in implementation of the exchange functional
//   cf_impl    : If we has built-in implementation of the correlation functional
//   npt        : Number of points
//   rho        : Size npt, electron density at some integral points
//   ipw        : Size npt, numerical integral weights of points in rho
//   workbuf    : Size npt * 4
//   p_libxc_xf : Pointer to Libxc exchange functional handle
//   p_libxc_xf : Pointer to Libxc correlation functional handle
// Output paramaters:
//   exc      : Size npt, = G / rho
//   vxc      : Size npt, = \frac{\part G}{\part rho}
//   <return> : XC energy, = \int G(rho(r)) dr
static double TinyDFT_eval_LDA_XC_func(
    const int xf_id, const int cf_id, const int xf_impl, const int cf_impl,
    const int npt, const double *rho, const double *ipw, double *workbuf, 
    xc_func_type *p_libxc_xf, xc_func_type *p_libxc_cf, double *exc, double *vxc
)
{
    double *ex = workbuf + npt * 0;
    double *ec = workbuf + npt * 1;
    double *vx = workbuf + npt * 2;
    double *vc = workbuf + npt * 3;

    if (xf_impl == 1)
    {
        eval_LDA_exc_vxc(xf_id, npt, rho, ex, vx);
    } else {
        xc_lda_exc_vxc(p_libxc_xf, npt, rho, ex, vx);
    }
    
    if (cf_impl == 1)
    {
        eval_LDA_exc_vxc(cf_id, npt, rho, ec, vc);
    } else {
        xc_lda_exc_vxc(p_libxc_cf, npt, rho, ec, vc);
    }
    
    double E_xc = 0.0;
    #pragma omp simd
    for (int i = 0; i < npt; i++)
    {
        exc[i] = ex[i] + ec[i];
        vxc[i] = vx[i] + vc[i];
        E_xc += exc[i] * rho[i] * ipw[i];
    }
    
    return E_xc;
}

// Build partial DFT XC matrix using LDA functional and 
// accumulate it to the final XC matrix
// Input parameters:
//   nbf     : Number of basis functions
//   npt     : The first npt phi values in phi will be used
//   ld_phi  : Leading dimension of phi, == maximum number of grid point 
//             results per basis function that phi can store
//   phi     : Size 4*nbf-by-ld_phi. In each nbf-by-ld_phi block,
//             the i-th row is the i-th basis function values (its 1st order 
//             derivatives on x, y, z directions) at some integral points
//   vxc     : Size npt, \frac{\part G}{\part rho}, will be multiplied by ipw
//   ipw     : Size npt, numerical integral weights of points in rho
//   beta    : 0.0 if this is the first call, otherwise 1.0
//   workbuf : Size npt * (nbf + 4)
// Output parameter:
//   XC_mat : Accumulated DFT XC matrix
static void TinyDFT_build_XC_LDA_partial(
    const int nbf, const int npt, const int ld_phi, 
    const double *phi, double *vxc, const double *ipw,
    const double beta, double *workbuf, double *XC_mat
)
{
    // XC_{u,v} = \sum_{g} phi_{u,g} * vxc_{g} * ipw_{g} * phi_{v,g} 
    double *phi_vxc_w = workbuf + npt * 4;
    #pragma omp simd
    for (int g = 0; g < npt; g++) vxc[g] *= ipw[g];
    #pragma omp parallel for
    for (int u = 0; u < nbf; u++)
    {
        const double *phi_u = phi + u * ld_phi;
        double *phi_vxc_w_u = phi_vxc_w + u * npt;
        #pragma omp simd
        for (int g = 0; g < npt; g++)
            phi_vxc_w_u[g] = phi_u[g] * vxc[g];
    }
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, nbf, nbf, npt,
        1.0, phi, ld_phi, phi_vxc_w, npt, beta, XC_mat, nbf
    );
}


// Evaluate GGA XC functional and calculate XC energy
// Input parameters:
//   xf_id      : Exchange functional ID
//   cf_id      : Correlation functional ID
//   xf_impl    : If we has built-in implementation of the exchange functional
//   cf_impl    : If we has built-in implementation of the correlation functional
//   npt        : Number of points
//   rho        : Size npt, electron density at some integral points
//   sigma      : Size npt, contracted gradient of rho
//   ipw        : Size npt, numerical integral weights of points in rho
//   workbuf    : Size npt * 6
//   p_libxc_xf : Pointer to Libxc exchange functional handle
//   p_libxc_xf : Pointer to Libxc correlation functional handle
// Output paramaters:
//   exc      : Size npt, = G / rho
//   vrho     : Size npt, = \frac{\part G}{\part rho}
//   vsigma   : Size npt, = \frac{\part G}{\part sigma}
//   <return> : XC energy, = \int G(rho(r)) dr
static double TinyDFT_eval_GGA_XC_func(
    const int xf_id, const int cf_id, const int xf_impl, const int cf_impl,
    const int npt, double *rho, const double *sigma, const double *ipw, 
    double *workbuf, xc_func_type *p_libxc_xf, xc_func_type *p_libxc_cf, 
    double *exc, double *vrho, double *vsigma
)
{
    double *ex      = workbuf + npt * 0;
    double *ec      = workbuf + npt * 1;
    double *vrhox   = workbuf + npt * 2;
    double *vrhoc   = workbuf + npt * 3;
    double *vsigmax = workbuf + npt * 4;
    double *vsigmac = workbuf + npt * 5;

    // Avoid rho[i] == 0
    for (int i = 0; i < npt; i++) rho[i] += 1e-60;

    if (xf_impl == 1)
    {
        eval_GGA_exc_vxc(xf_id, npt, rho, sigma, ex, vrhox, vsigmax);
    } else {
        xc_gga_exc_vxc(p_libxc_xf, npt, rho, sigma, ex, vrhox, vsigmax);
    }
    
    if (cf_impl == 1)
    {
        eval_GGA_exc_vxc(cf_id, npt, rho, sigma, ec, vrhoc, vsigmac);
    } else {
        xc_gga_exc_vxc(p_libxc_cf, npt, rho, sigma, ec, vrhoc, vsigmac);
    }
    
    double E_xc = 0.0;
    #pragma omp simd
    for (int i = 0; i < npt; i++)
    {
        exc[i]    = ex[i]      + ec[i];
        vrho[i]   = vrhox[i]   + vrhoc[i];
        vsigma[i] = vsigmax[i] + vsigmac[i];  // source of bug??
        E_xc += exc[i] * rho[i] * ipw[i];
    }
    
    return E_xc;
}

// Build partial DFT XC matrix using GGA functional and 
// accumulate it to the final XC matrix
// Input parameters:
//   nbf     : Number of basis functions
//   npt     : The first npt phi values in phi will be used
//   ld_phi  : Leading dimension of phi, == maximum number of grid point 
//             results per basis function that phi can store
//   phi     : Size 4*nbf-by-ld_phi. In each nbf-by-ld_phi block,
//             the i-th row is the i-th basis function values (its 1st order 
//             derivatives on x, y, z directions) at some integral points
//   ld_rho  : Leading dimension of rho (should == ld_phi)
//   rho     : Size 5*nintp_blk, electron density, its 1st order derivatives
//             on x, y, z directions, and its contracted gradient (sigma)
//             at some integral points
//   vrho    : Size npt, \frac{\part G}{\part rho}, will be multiplied by ipw
//   vsigma  : Size npt, \frac{\part G}{\part sigma}, will be multiplied by 2*ipw
//   ipw     : Size npt, numerical integral weights of points in rho
//   beta    : 0.0 if this is the first call, otherwise 1.0
//   workbuf : Size npt * (nbf + 6)
// Output parameter:
//   XC_mat : Accumulated DFT XC matrix
static void TinyDFT_build_XC_GGA_partial(
    const int nbf, const int npt, const int ld_phi, const double *phi, 
    const int ld_rho, const double *rho, double *vrho, double *vsigma, 
    const double *ipw, const double beta, double *workbuf, double *XC_mat
)
{
    // 1. \sum_{g} phi_{u,g} * vrho_{g} * ipw_{g} * phi_{v,g} 
    double *phi_vrho_w = workbuf + npt * 6;
    #pragma omp simd
    for (int g = 0; g < npt; g++) vrho[g] *= ipw[g];
    #pragma omp parallel for
    for (int u = 0; u < nbf; u++)
    {
        const double *phi_u = phi + u * ld_phi;
        double *phi_vrho_w_u = phi_vrho_w + u * npt;
        #pragma omp simd
        for (int g = 0; g < npt; g++)
            phi_vrho_w_u[g] = phi_u[g] * vrho[g];
    }
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, nbf, nbf, npt,
        1.0, phi, ld_phi, phi_vrho_w, npt, beta, XC_mat, nbf
    );
    
    // 2.1 \sum_{g} ipw_{g} * 2 * vsigma_{g} * [drho_dx_{g} * (phi_{u,g} * dphi_dx_{v,g} + phi_{v,g} * dphi_dx_{u,g})]
    // 2.2 \sum_{g} ipw_{g} * 2 * vsigma_{g} * [drho_dy_{g} * (phi_{u,g} * dphi_dy_{v,g} + phi_{v,g} * dphi_dy_{u,g})]
    // 2.3 \sum_{g} ipw_{g} * 2 * vsigma_{g} * [drho_dz_{g} * (phi_{u,g} * dphi_dz_{v,g} + phi_{v,g} * dphi_dz_{u,g})]
    const double *dphi_dx = phi + nbf * ld_phi * 1;
    const double *dphi_dy = phi + nbf * ld_phi * 2;
    const double *dphi_dz = phi + nbf * ld_phi * 3;
    const double *drho_dx = rho + ld_rho * 1;
    const double *drho_dy = rho + ld_rho * 2;
    const double *drho_dz = rho + ld_rho * 3;
    #pragma omp simd
    for (int g = 0; g < npt; g++) vsigma[g] *= 2.0 * ipw[g];
    #pragma omp parallel for
    for (int u = 0; u < nbf; u++)
    {
        int phi_offset_u = u * ld_phi;
        const double *phi_u     = phi     + phi_offset_u;
        const double *dphi_dx_u = dphi_dx + phi_offset_u;
        const double *dphi_dy_u = dphi_dy + phi_offset_u;
        const double *dphi_dz_u = dphi_dz + phi_offset_u;
        for (int v = 0; v < nbf; v++)
        {
            int phi_offset_v = v * ld_phi;
            const double *phi_v     = phi     + phi_offset_v;
            const double *dphi_dx_v = dphi_dx + phi_offset_v;
            const double *dphi_dy_v = dphi_dy + phi_offset_v;
            const double *dphi_dz_v = dphi_dz + phi_offset_v;
            
            double res0 = 0.0, res1 = 0.0, res2 = 0.0;
            #pragma omp simd 
            for (int g = 0; g < npt; g++)
            {
                res0 += vsigma[g] * drho_dx[g] * (phi_u[g] * dphi_dx_v[g] + phi_v[g] * dphi_dx_u[g]);
                res1 += vsigma[g] * drho_dy[g] * (phi_u[g] * dphi_dy_v[g] + phi_v[g] * dphi_dy_u[g]);
                res2 += vsigma[g] * drho_dz[g] * (phi_u[g] * dphi_dz_v[g] + phi_v[g] * dphi_dz_u[g]);
            }
            XC_mat[u * nbf + v] += res0 + res1 + res2;
        }
    }
}

// Construct DFT exchange-correlation matrix
double TinyDFT_build_XC_mat(TinyDFT_t TinyDFT, const double *D_mat, double *XC_mat)
{ 
    int    nbf        = TinyDFT->nbf;
    int    xf_id      = TinyDFT->xf_id;
    int    cf_id      = TinyDFT->cf_id;
    int    xf_impl    = TinyDFT->xf_impl;
    int    cf_impl    = TinyDFT->cf_impl;
    int    xf_family  = TinyDFT->xf_family;
    int    cf_family  = TinyDFT->cf_family;
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
    double *vrho      = TinyDFT->vxc;
    double *vsigma    = TinyDFT->vsigma;
    double *int_grid  = TinyDFT->int_grid;
    double *ipw       = TinyDFT->int_grid + 3 * nintp;
    double *workbuf   = TinyDFT->XC_workbuf;
    
    xc_func_type *p_libxc_xf = &TinyDFT->libxc_xf;
    xc_func_type *p_libxc_cf = &TinyDFT->libxc_cf;

    double E_xc = 0.0;
    for (int sintp = 0; sintp < nintp; sintp += nintp_blk)
    {
        int eintp = sintp + nintp_blk;
        if (eintp > nintp) eintp = nintp;
        int npt = eintp - sintp;
        double beta = (sintp == 0) ? 0.0 : 1.0;
        double *curr_ipw = ipw + sintp;
        
        TinySCF_eval_basis_func(
            nintp, int_grid, sintp, eintp, 
            nbf, bf_coef, bf_alpha,
            bf_exp, bf_center, bf_nprim, 
            max_nprim, nintp_blk, phi
        );
        
        TinyDFT_eval_electron_density(
            nbf, D_mat, npt, nintp_blk, 
            phi, nintp_blk, rho, workbuf
        );
        
        if (cf_family == FAMILY_LDA)
        {
            E_xc += TinyDFT_eval_LDA_XC_func(
                xf_id, cf_id, xf_impl, cf_impl, 
                npt, rho, curr_ipw, workbuf, 
                p_libxc_xf, p_libxc_cf, exc, vxc
            );
            
            TinyDFT_build_XC_LDA_partial(
                nbf, npt, nintp_blk, phi, vxc, 
                curr_ipw, beta, workbuf, XC_mat
            );
        }
        
        if (cf_family == FAMILY_GGA)
        {
            E_xc += TinyDFT_eval_GGA_XC_func(
                xf_id, cf_id, xf_impl, cf_impl, 
                npt, rho, vsigma, curr_ipw, 
                workbuf, p_libxc_xf, p_libxc_cf, 
                exc, vrho, vsigma
            );
            
            TinyDFT_build_XC_GGA_partial(
                nbf, npt, nintp_blk, phi, 
                nintp_blk, rho, vrho, vsigma, 
                curr_ipw, beta, workbuf, XC_mat
            );
        }
    }
    
    printf("E_xc = %lf\n", E_xc);
    return E_xc;
}

