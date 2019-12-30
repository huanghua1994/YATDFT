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
void eval_LDA_exc_vxc_C_XA(const int npt, const double *rho, double *exc, double *vxc)
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
void eval_LDA_exc_vxc_C_PZ81(const int npt, const double *rho, double *exc, double *vxc)
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
void eval_LDA_exc_vxc_C_PW92(const int npt, const double *rho, double *exc, double *vxc)
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
        case LDA_X:    eval_LDA_exc_vxc_X     (npt, rho, exc, vxc); break;
        case LDA_C_XA: eval_LDA_exc_vxc_C_XA  (npt, rho, exc, vxc); break;
        case LDA_C_PZ: eval_LDA_exc_vxc_C_PZ81(npt, rho, exc, vxc); break;
        case LDA_C_PW: eval_LDA_exc_vxc_C_PW92(npt, rho, exc, vxc); break;
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

// Perdew, Burke & Ernzerhof exchange
// Based on Libxc implementation, manually optimized
static void eval_GGA_exc_vxc_X_PBE(
    const int npt, const double *rho, const double *sigma, 
    double *exc, double *vrho, double *vsigma
)
{
    const double M_CBRT2 = 1.25992104989487316476721060727;
    const double M_CBRT3 = 1.44224957030740838232163831078;
    const double M_CBRT4 = 1.58740105196819947475170563927;
    const double M_CBRT6 = 1.81712059283213965889121175632;
    const double kappa   = 0.8040;
    const double mu      = 0.2195149727645171;  // mu = beta*pi^2/3
    const double PI_N1_3 = 0.68278406325529568146702083315;  // cbrt(1.0 / M_PI)
    
    double r24 = 1.0 / 24.0;
    double t6  = M_CBRT4 * M_CBRT4;
    double t9  = M_CBRT2 * M_CBRT2;
    double t7  = M_CBRT3 * PI_N1_3 * t6 * t9;
    double t15 = cbrt(M_PI * M_PI);
    double t17 = M_CBRT6 / (t15 * t15);
    double t18 = mu * t17;
    double t19 = t9 * t18 * r24;
    double t41 = PI_N1_3 * M_CBRT2 * M_CBRT3 * t6;
    double t43 = kappa * kappa;
    t18 *= 0.015625;
    
    #pragma omp simd
    for (int i = 0; i < npt; i++)
    {
        double rho_1_3  = cbrt(rho[i]) + 1e-30;
        double rho_n1_3 = 1.0 / rho_1_3;
        double rho_n2_3 = rho_n1_3 * rho_n1_3;
        double inv_rho  = rho_n2_3 * rho_n1_3;
        double inv_rho2 = inv_rho  * inv_rho;

        double t27 = 1.0 / (kappa + sigma[i] * rho_n2_3 * inv_rho2 * t19);
        double t34 = -0.25 * rho_1_3 * t7 * (1.0 + kappa - t27 * t43);
        double t46 = t27 * t27 * t43;
        double t50 = sigma[i] * mu * t17 * t46;
        
        exc[i]    = 0.75 * t34;
        vrho[i]   = t34 + rho_n1_3 * inv_rho2 * t41 * t50 * r24;
        vsigma[i] = -rho_n1_3 * inv_rho * t18 * t41 * t46;
    }
}

// Perdew, Burke & Ernzerhof correlation
// Based on Libxc implementation, manually optimized
static void eval_GGA_exc_vxc_C_PBE(
    const int npt, const double *rho, const double *sigma, 
    double *exc, double *vrho, double *vsigma
)
{
    const double M_CBRT2 = 1.25992104989487316476721060727;
    const double M_CBRT3 = 1.44224957030740838232163831078;
    const double M_CBRT4 = 1.58740105196819947475170563927;
    const double beta    = 0.06672455060314922;
    const double gamma   = 0.031090690869654895034;
    const double BB      = 1.0;
    const double PI_N1_3 = 0.68278406325529568146702083315;  // cbrt(1.0 / M_PI)
    
    double t4   = M_CBRT3 * PI_N1_3;
    double t6   = M_CBRT4 * M_CBRT4;
    double t18  = M_CBRT3 * M_CBRT3;
    double t19  = PI_N1_3 * PI_N1_3;
    double t20  = t18 * t19 * M_CBRT4;
    double t41  = M_CBRT2 * t18 / PI_N1_3 * M_CBRT4;
    double t44  = BB * beta;
    double t45  = 1.0 / gamma;
    double t46  = t44 * t45;
    double t58  = M_CBRT2 * M_CBRT2;
    double t60  = 1.0 / t19;
    double t64  = t41 / 96.0;
    double t122 = t58 * M_CBRT3;
    double t124 = t4 * t6;
    double t177 = beta * beta;
    double t180 = 1.0 / (gamma * gamma);
    
    #pragma omp simd
    for (int i = 0; i < npt; i++)
    {
        double rho_n1_3 = 1.0 / cbrt(rho[i]);
        double rho_n2_3 = rho_n1_3 * rho_n1_3;
        double inv_rho  = rho_n2_3 * rho_n1_3;
        double rho_n4_3 = rho_n1_3 * inv_rho;
        double inv_rho2 = inv_rho  * inv_rho;
        double rho_n7_3 = rho_n1_3 * inv_rho2;
        double inv_rho3 = inv_rho2 * inv_rho;
        double rho_2    = rho[i] * rho[i];
        double rho_4    = rho_2  * rho_2;
        
        double t7  = t6 * rho_n4_3;
        double t10 = t124 * rho_n1_3;
        double t12 = 1.0 + 0.053425 * t10;
        double t13 = sqrt(t10);
        double t26 = 3.79785  * t13 + 
                     0.8969   * t10 + 
                     0.204775 * t10 * t13 + 
                     0.123235 * t20 * rho_n2_3;
        double t27 = 1.0 / t26;
        double t29 = 1.0 + 16.081979498692535067 * t27;
        double t30 = log(t29);
        double t31 = t12 * t30;
        double t32 = 0.0621814 * t31;
        double t48 = exp(0.0621814 * t31 * t45);
        double t50 = 1.0 / (t48 - 1.0);
        double t51 = t46 * t50;
        double t52 = sigma[i] * sigma[i];
        double t54 = t51 * t52;
        double t57 = rho_n7_3 * rho_n7_3;
        double t62 = M_CBRT3 * t6 * t60;
        double t63 = t57 * t58 * t62 * 6.510416666666666e-04;
        double t66 = sigma[i] * rho_n7_3 * t64 + t54 * t63 * 0.5;
        double t67 = beta * t66;
        double t68 = beta * t45;
        double t69 = t68 * t50;
        double t71 = 1.0 / (t69 * t66 + 1.0);
        double t73 = t45 * t71;
        double t75 = t67 * t73 + 1.0;
        double t76 = log(t75);
        double t77 = gamma * t76;
        exc[i] = t77 - t32;
        
        double t80  = t4 * t7;
        double t86  = t12 * t27 * t27;
        double t96  = PI_N1_3 * M_CBRT3 * t7;
        double t104 = -0.632975 * t96 / t13 - 0.29896666666666666667 * t80 - 0.1023875 * t96 * t13 - 0.082156666666666666667 * t20 * rho_n2_3 * inv_rho;
        double t106 = t104 / t29;
        double t132 = -0.0011073470983333333333 * rho_n4_3 * t30 * t45 * t124 - t45 * t86 * t106;
        double t118 = t50 * t50 * t132;
        double t145 = -(7.0/288.0) * sigma[i] * rho_n1_3 * inv_rho3 * t41 
                      -t6 * t46 * t48 * t52 * t57 * t60 * t118 * t122 / 3072.0 
                      -(7.0/4608.0) * rho_n2_3 * inv_rho2 * inv_rho3 * t54 * t58 * t62;
        double t149 = t71 * t71;
        double t157 = t69 * t145 - t48 * t66 * t68 * t118;
        double t160 = beta * t73 * t145 - t67 * t45 * t149 * t157;
        double t162 = 1.0 / t75;
        vrho[i] = t77 - t32 + rho[i] * (0.0011073470983333333333 * t30 * t80 + t86 * t106 + gamma * t160 * t162);

        double t174 = rho_n7_3 * t64 + sigma[i] * t51 * t63;
        double t185 = t174 * (beta * t73 - t50 * t66 * t149 * t177 * t180);
        vsigma[i] = rho[i] * gamma * t185 * t162;
        
        if (rho[i] < 1e-12)
        {
            exc[i]    = 0.0;
            vrho[i]   = 0.0;
            vsigma[i] = 0.0;
        }
    }
}

// Evaluate GGA XC functional E_xc = \int G(rho(r)) dr
void eval_GGA_exc_vxc(
    const int fid, const int npt, const double *rho, const double *sigma, 
    double *exc, double *vrho, double *vsigma
)
{
    switch (fid)
    {
        case GGA_X_PBE:  eval_GGA_exc_vxc_X_PBE(npt, rho, sigma, exc, vrho, vsigma); break;
        case GGA_C_PBE:  eval_GGA_exc_vxc_C_PBE(npt, rho, sigma, exc, vrho, vsigma); break;
    }
}

