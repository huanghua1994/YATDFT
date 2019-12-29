#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "eval_XC_func.h"

// ========================= LDA functionals ========================= //
// Input parameters:
//   npt : Total number of points
//   rho : Size npt, electron density at each point
// Output parameters:
//   exc : Size npt, = G / rho
//   vxc : Size npt, = \frac{\part G}{\part rho}

// Slater exchange
void eval_LDA_exc_vxc_X(const int npt, const double *rho, double *exc, double *vxc)
{
    const double c0 = 0.9847450218426965; // (3/pi)^(1/3)
    const double a  = 2.0/3.0;
    const double c1 = -a * (9.0/8.0) * c0;
    const double c2 = -a * (3.0/2.0) * c0;
    #pragma omp simd
    for (int i = 0; i < npt; i++)
    {
        double rho_cbrt = cbrt(rho[i]);
        exc[i] = c1 * rho_cbrt;
        vxc[i] = c2 * rho_cbrt;
    }
}

// Slater Xalpha correlation (Xalpha = 0.7 - 2/3)
void eval_LDA_exc_vxc_XA(const int npt, const double *rho, double *exc, double *vxc)
{
    const double c0 = 0.9847450218426965;   // (3/pi)^(1/3)
    const double a  = 0.7 - (2.0/3.0);      // alpha = 0.7, minus 2/3 exchange part
    const double c1 = -a * (9.0/8.0) * c0;
    const double c2 = -a * (3.0/2.0) * c0;
    #pragma omp simd
    for (int i = 0; i < npt; i++)
    {
        double rho_cbrt = cbrt(rho[i]);
        exc[i] = c1 * rho_cbrt;
        vxc[i] = c2 * rho_cbrt;
    }
}

// LDA Perdew-Zunger correlation (PZ81)
void eval_LDA_exc_vxc_PZ81(const int npt, const double *rho, double *exc, double *vxc)
{
    // PZ81 correlation parameters
    const double A  =  0.0311;
    const double B  = -0.048;
    const double C  =  0.002;
    const double D  = -0.0116;
    const double g1 = -0.1423;  // gamma1
    const double b1 =  1.0529;  // beta1
    const double b2 =  0.3334;  // beta2
    
    const double c0 = 0.6203504908993999;  // (3/(4*pi))^(1/3)
    
    const double t0 = 2.0 * C / 3.0;
    const double t1 = B - A / 3.0;
    const double t2 = (2.0 * D - C) / 3.0;
    const double t3 = (7.0/6.0) * g1 * b1;
    const double t4 = (4.0/3.0) * g1 * b2;
    
    #pragma omp simd
    for (int i = 0; i < npt; i++)
    {
        double rho_cbrt = cbrt(rho[i]) + 1e-20;
        double rs = c0 / rho_cbrt;  // rs = (3/(4*pi*rho))^(1/3)
        if (rs < 1.0)
        {
            double log_rs = log(rs);
            vxc[i] = log_rs * (A + t0 * rs) + t1 + t2 * rs;
            exc[i] = log_rs * (A +  C * rs) +  B +  D * rs;
        } else {
            double rs_05 = sqrt(rs);
            double t6 = 1.0 / (1.0 + b1 * rs_05 + b2 * rs);
            vxc[i] = (g1 + t3 * rs_05 + t4 * rs) * t6 * t6;
            exc[i] = g1 * t6;
        }
    }
}

// LDA Perdew-Wang correlation (PW92)
void eval_LDA_exc_vxc_PW92(const int npt, const double *rho, double *exc, double *vxc)
{
    // PW92 correlation parameters
    const double p  = 1.0;
    const double A  = 0.031091;
    const double A2 = 0.062182; // A * 2
    const double a1 = 0.21370;  // alpha1
    const double b1 = 7.5957;   // beta1
    const double b2 = 3.5876;   // beta2
    const double b3 = 1.6382;   // beta3
    const double b4 = 0.49294;  // beta4
    
    const double c0 = 0.6203504908993999;  // (3/(4*pi))^(1/3)
    
    const double t0 = 2.0 * b2;
    const double t1 = 3.0 * b3;
    const double t2 = 2.0 * (p+1.0) * b4;
    
    #pragma omp simd
    for (int i = 0; i < npt; i++)
    {
        double rho_cbrt = cbrt(rho[i]) + 1e-20;
        double rs    = c0 / rho_cbrt;   // rs = (3/(4*pi*rho))^(1/3)
        double rs_05 = sqrt(rs);        // rs^0.5
        double rs_15 = rs_05 * rs;      // rs^1.5
        double rs_p  = rs;              // rs^p, p = 1
        double rs_p1 = rs_p * rs;       // rs^(p+1)
        double G2    = A2 * (b1*rs_05 + b2*rs + b3*rs_15 + b4*rs_p1);
        double G1    = log(1.0 + 1.0 / G2);
        
        double vxc0 = A2 * (1.0 + a1*rs);
        double vxc1 = -vxc0 * G1;
        double vxc2 = -A2 * a1 * G1;
        double vxc3 = b1/rs_05 + t0 + t1*rs_05 + t2*rs_p;
        exc[i] = vxc1;
        vxc[i] = vxc1 - (rs/3.0) * (vxc2 + (vxc0 * (A * vxc3)) / (G2 * (G2+1.0)) );
    }
}

// Evaluate LDA XC functional E_xc = \int G(rho(r)) dr
void eval_LDA_exc_vxc(const int fid, const int npt, const double *rho, double *exc, double *vxc)
{
    switch (fid)
    {
        case LDA_X:    eval_LDA_exc_vxc_X   (npt, rho, exc, vxc); break;
        case LDA_C_XA: eval_LDA_exc_vxc_XA  (npt, rho, exc, vxc); break;
        case LDA_C_PZ: eval_LDA_exc_vxc_PZ81(npt, rho, exc, vxc); break;
        case LDA_C_PW: eval_LDA_exc_vxc_PW92(npt, rho, exc, vxc); break;
    }
}

// ========================= GGA functionals ========================= //
// Input parameters:
//   npt   : Total number of points
//   rho   : Size npt, electron density at each point
//   sigma : Size npt, contracted gradient of rho
// Output parameters:
//   exc    : Size npt, = G / rho
//   vrho   : Size npt, = \frac{\part G}{\part rho}
//   vsigma : Size npt, = \frac{\part G}{\part sigma}

// Evaluate GGA XC functional E_xc = \int G(rho(r)) dr
void eval_GGA_exc_vxc(
    const int fid, const int npt, const double *rho, const double *sigma, 
    double *exc, double *vrho, double *vsigma
)
{
    // To be done?
}
