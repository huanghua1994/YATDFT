#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "gen_Lebedev_grid.h"

const static double BOHR = 0.52917721092;

const static int Leb_ngrid[33] = {
       1,    6,   14,   26,   38,   50,   74,   86,  110,  146,
     170,  194,  230,  266,  302,  350,  434,  590,  770,  974,
    1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334,
    4802, 5294, 5810
};

// Need to multiple by (1.0 / BOHR) to get the real RADII_BRAGG
const static double RADII_BRAGG[] = {
    0.35,                                     1.40,              // 1s
    1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50,              // 2s2p
    1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80,              // 3s3p
    2.20, 1.80,                                                  // 4s
    1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35,  // 3d
                1.30, 1.25, 1.15, 1.15, 1.15, 1.90,              // 4p
    2.35, 2.00,                                                  // 5s
    1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55,  // 4d
                1.55, 1.45, 1.45, 1.40, 1.40, 2.10,              // 5p
    2.60, 2.15,                                                  // 6s
    1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85,                    // La, Ce-Eu
    1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,              // Gd, Tb-Lu
          1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35, 1.50,  // 5d
                1.90, 1.80, 1.60, 1.90, 1.45, 2.10,              // 6p
    1.80, 2.15,                                                  // 7s
    1.95, 1.80, 1.80, 1.75, 1.75, 1.75, 1.75,                   
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,                   
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 
                1.75, 1.75, 1.75, 1.75, 1.75, 1.75,             
    1.75, 1.75,                                                 
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75  
};

static double alphas[3][4] = {
    {0.25,    0.5, 1.0, 4.5},
    {0.16667, 0.5, 0.9, 3.5},
    {0.1,     0.4, 0.8, 2.5}
};

// Generate Becke grid points for radial integral
// Ref: [JCP 88, 2547], DOI: 10.1063/1.454033
// Input parameters:
//   npt2 : Total number of grid points+2
//   rm   : A parameter mentioned in Ref, directly use 1.0 is fine
// Output parameters:
//   rad_r : Size npt2-2, radial direction integral points
//   rad_w : Size npt2-2, radial direction integral weights
static void gen_Becke_grid(const int npt2, const double rm, double *rad_r, double *rad_w)
{
    int npt1 = npt2 - 1;
    int npt  = npt2 - 2;
    double pi_over_npt1 = M_PI / (double) npt1;
    double rm3 = rm * rm * rm;
    // Skip i = 0, since rad_r(0) = rad_w(0) = inf
    // Skip i = npt1, since rad_r(npt1) = rad_w(npt1) = 0
    #pragma omp simd
    for (int i = 1; i <= npt; i++)
    {
        // radr and radw formulas come from: http://sobereva.com/69 ,
        // or multiwfn v3.7 fuzzy.f90, line 449
        double radx = cos((double) i * pi_over_npt1);
        double radr = rm * (1.0 + radx) / (1.0 - radx);
        double tmp0 = pow(1.0 + radx, 2.5);
        double tmp1 = pow(1.0 - radx, 3.5);
        double radw = (2.0 * pi_over_npt1) * rm3 * tmp0 / tmp1;
        rad_r[npt - i] = radr;
        rad_w[npt - i] = radw;
    }
}

// Prune grids using NWChem scheme
// Ref: 
//   1. https://github.com/pyscf/pyscf/blob/master/pyscf/dft/gen_grid.py
//   2. https://github.com/pyscf/pyscf/blob/master/pyscf/data/radii.py
// Input parameters:
//   nuc   : Nuclear charge (== atom index)
//   n_ang : Maximum number of angular grids
//   n_rad : Number of radial grids
//   rads  : Array, size n_rad, radial grid coordinates
// Output parameter:
//   rad_n_ang : Array, size n_rad, number of angular grids for each radial grid
//   <return>  : Total number of grid points, == sum(rad_n_ang)
static int NWChem_prune_grid(
    const int nuc, const int n_ang, const int n_rad, 
    const double *rads, int *rad_n_ang
)
{
    int Leb_lvl[5] = {4, 5, 5, 5, 4};
    if (n_ang < 50)
    {
        for (int i = 0; i < n_rad; i++) rad_n_ang[i] = n_ang;
        return (n_ang * n_rad);
    }
    if (n_ang > 50)
    {
        int idx;
        for (idx = 6; idx < 33; idx++)
            if (n_ang == Leb_ngrid[idx]) break;
        Leb_lvl[1] = 6;
        Leb_lvl[2] = idx - 1;
        Leb_lvl[3] = idx;
        Leb_lvl[4] = idx - 1;
    }
    int npt = 0;
    double r_atom = (1.0 / BOHR) * RADII_BRAGG[nuc - 1];
    for (int i = 0; i < n_rad; i++)
    {
        double rad_i_scale = rads[i] / r_atom;
        double *alpha_i;
        if (nuc >  10) alpha_i = alphas[2];
        if (nuc <= 10) alpha_i = alphas[1];
        if (nuc <=  2) alpha_i = alphas[0];
        int place;
        for (place = 0; place < 4; place++)
            if (rad_i_scale <= alpha_i[place]) break;
        rad_n_ang[i] = Leb_ngrid[Leb_lvl[place]];
        npt += rad_n_ang[i];
    }
    return npt;
}

// Generate numerical integral points for XC calculations
void gen_int_grid(
    const int natom, const double *atom_xyz, const int *atom_idx,
    int *npoint_, double **int_grid_
)
{
    double rm = 1.0;
    int max_rad = 65;
    int max_ang = 302;
    int max_atom_npt  = max_rad * max_ang;
    int max_total_npt = natom * max_atom_npt;

    double *int_grid = (double*) malloc(sizeof(double) * 4 * max_total_npt);
    double *dist     = (double*) malloc(sizeof(double) * natom * natom);
    assert(int_grid != NULL && dist != NULL);
    double *ipx = int_grid + 0 * max_total_npt;
    double *ipy = int_grid + 1 * max_total_npt;
    double *ipz = int_grid + 2 * max_total_npt;
    double *ipw = int_grid + 3 * max_total_npt;
    
    const double *atom_x = atom_xyz + 0 * natom;
    const double *atom_y = atom_xyz + 1 * natom;
    const double *atom_z = atom_xyz + 2 * natom;
    for (int i = 0; i < natom; i++)
    {
        #pragma omp simd
        for (int j = 0; j < natom; j++)
        {
            double dx = atom_x[i] - atom_x[j];
            double dy = atom_y[i] - atom_y[j];
            double dz = atom_z[i] - atom_z[j];
            double r2 = dx * dx + dy * dy + dz * dz;
            dist[i * natom + j] = sqrt(r2);
        }
    }
    
    double *rad_r     = (double*) malloc(sizeof(double) * max_rad);
    double *rad_w     = (double*) malloc(sizeof(double) * max_rad);
    int    *rad_n_ang = (int*)    malloc(sizeof(int)    * max_rad);
    double *leb_tmp   = (double*) malloc(sizeof(double) * max_ang * 4);
    double *atom_grid = (double*) malloc(sizeof(double) * max_atom_npt * 4);
    double *W_mat     = (double*) malloc(sizeof(double) * natom * natom * max_atom_npt);
    double *dip       = (double*) malloc(sizeof(double) * natom * max_atom_npt);
    double *pvec      = (double*) malloc(sizeof(double) * natom * max_atom_npt);
    double *sum_pvec  = (double*) malloc(sizeof(double) * max_atom_npt);
    assert(rad_r   != NULL && rad_w != NULL && rad_n_ang != NULL);
    assert(leb_tmp != NULL && W_mat != NULL && atom_grid != NULL);
    assert(dip     != NULL && pvec  != NULL && sum_pvec  != NULL);
    int total_npt = 0;
    for (int iatom = 0; iatom < natom; iatom++)
    {
        // (1) Prune grid points according to atom type
        int n_ang = max_ang;
        int n_rad = max_rad;
        if (atom_idx[iatom] <= 10) n_rad = 50;
        if (atom_idx[iatom] <=  2) n_rad = 35;
        gen_Becke_grid(n_rad + 2, rm, rad_r, rad_w);
        int atom_npt = NWChem_prune_grid(atom_idx[iatom], n_ang, n_rad, rad_r, rad_n_ang);
        
        // (2) Generate Lebedev points & weights and combine it
        //     with radial direction points & weights
        int atom_idx = 0;
        double *atom_ipx = atom_grid + 0 * atom_npt;
        double *atom_ipy = atom_grid + 1 * atom_npt;
        double *atom_ipz = atom_grid + 2 * atom_npt;
        double *atom_ipw = atom_grid + 3 * atom_npt;
        for (int i = 0; i < n_rad; i++)
        {
            int npt_i = gen_Lebedev_grid(rad_n_ang[i], leb_tmp);
            #pragma omp simd
            for (int j = 0; j < rad_n_ang[i]; j++)
            {
                atom_ipx[atom_idx + j] = leb_tmp[j * 4 + 0] * rad_r[i];
                atom_ipy[atom_idx + j] = leb_tmp[j * 4 + 1] * rad_r[i];
                atom_ipz[atom_idx + j] = leb_tmp[j * 4 + 2] * rad_r[i];
                atom_ipw[atom_idx + j] = leb_tmp[j * 4 + 3] * rad_w[i] * 4 * M_PI;
                
                atom_ipx[atom_idx + j] += atom_x[iatom];
                atom_ipy[atom_idx + j] += atom_y[iatom];
                atom_ipz[atom_idx + j] += atom_z[iatom];
            }
            atom_idx += npt_i;
        }  // End of i loop
        
        // (3) Calculate the mask tensor and the actual weights
        // W_mat(j, k, i): fuzzy weight of integral point i to atom pair (j, k)
        for (int j = 0; j < natom; j++)
        {
            double *dip_j = dip + j * atom_npt;
            #pragma omp simd
            for (int i = 0; i < atom_npt; i++)
            {
                double dx = atom_ipx[i] - atom_x[j];
                double dy = atom_ipy[i] - atom_y[j];
                double dz = atom_ipz[i] - atom_z[j];
                dip_j[i] = sqrt(dx * dx + dy * dy + dz * dz);
            }
        }  // End of j loop
        // TODO: OpenMP parallelize this loop
        for (int j = 0; j < natom; j++)
        {
            for (int k = 0; k < natom; k++)
            {
                double *dip_j = dip + j * atom_npt;
                double *dip_k = dip + k * atom_npt;
                double *W_jk  = W_mat + (j * natom + k) * atom_npt;
                if (j == k)
                {
                    for (int i = 0; i < atom_npt; i++) W_jk[i] = 1.0;
                } else {
                    double inv_djk = 1.0 / dist[j * natom + k];
                    #pragma omp simd
                    for (int i = 0; i < atom_npt; i++)
                    {
                        double mu = (dip_j[i] - dip_k[i]) * inv_djk;
                        
                        // s(d(i,j)) = 0.5 * (1 - p(p(p(d(i,j)))))
                        mu = 1.5 * mu - 0.5 * mu * mu * mu;
                        mu = 1.5 * mu - 0.5 * mu * mu * mu;
                        mu = 1.5 * mu - 0.5 * mu * mu * mu;
                        
                        W_jk[i] = 0.5 * (1.0 - mu);
                    }
                }  // End of "if (j == k)"
            }  // End of k loop
        }  // End of j loop
        
        // (4) Calculate the final integral weights
        // \prod_{k} W_mat(j, k, :) is the actual weight of integral points
        // belonging to atom k. Normalizing it gives us the fuzzy weight.
        for (int i = 0; i < natom * atom_npt; i++) pvec[i] = 1.0;
        memset(sum_pvec, 0, sizeof(double) * atom_npt);
        // TODO: OpenMP parallelize this loop
        for (int j = 0; j < natom; j++)
        {
            double *pvec_j = pvec + j * atom_npt;
            for (int k = 0; k < natom; k++)
            {
                double *W_jk =  W_mat + (j * natom + k) * atom_npt;
                #pragma omp simd
                for (int i = 0; i < atom_npt; i++) pvec_j[i] *= W_jk[i];
            }
            for (int i = 0; i < atom_npt; i++) sum_pvec[i] += pvec_j[i];
        }
        // Copy final integral points & weights to the output matrix
        double *pvec_iatom = pvec + iatom * atom_npt;
        for (int i = 0; i < atom_npt; i++)
        {
            ipx[total_npt + i] = atom_ipx[i];
            ipy[total_npt + i] = atom_ipy[i];
            ipz[total_npt + i] = atom_ipz[i];
            ipw[total_npt + i] = atom_ipw[i] * pvec_iatom[i] / sum_pvec[i];
        }
        total_npt += atom_npt;
    }  // End of iatom loop
    
    double *new_int_grid = (double*) malloc(sizeof(double) * total_npt * 4);
    assert(new_int_grid != NULL);
    memcpy(new_int_grid + 0 * total_npt, ipx, sizeof(double) * total_npt);
    memcpy(new_int_grid + 1 * total_npt, ipy, sizeof(double) * total_npt);
    memcpy(new_int_grid + 2 * total_npt, ipz, sizeof(double) * total_npt);
    memcpy(new_int_grid + 3 * total_npt, ipw, sizeof(double) * total_npt);
    
    *npoint_   = total_npt;
    *int_grid_ = new_int_grid;
    free(int_grid);
    free(dist);
    free(rad_r);
    free(rad_w);
    free(rad_n_ang);
    free(leb_tmp);
    free(atom_grid);
    free(W_mat);
    free(dip);
    free(pvec);
    free(sum_pvec);
}

