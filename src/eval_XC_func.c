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
        double rho_1_3  = cbrt(rho[i]);
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
        
        if (rho[i] < 1e-12)
        {
            exc[i]    = 0.0;
            vrho[i]   = 0.0;
            vsigma[i] = 0.0;
        }
    }
}

// Becke 88 exchange
// Based on Libxc implementation, manually optimized
static void eval_GGA_exc_vxc_X_B88(
    const int npt, const double *rho, const double *sigma, 
    double *exc, double *vrho, double *vsigma
)
{
    const double M_CBRT2 = 1.25992104989487316476721060727;
    const double M_CBRT3 = 1.44224957030740838232163831078;
    const double M_CBRT4 = 1.58740105196819947475170563927;
    const double beta    = 0.0042;
    const double gamma   = 6.0;
    const double PI_N1_3 = 0.68278406325529568146702083315;  // cbrt(1.0 / M_PI)
    
    double t6   = M_CBRT4 * M_CBRT4;
    double t7   = M_CBRT3 * PI_N1_3 * t6;
    double t9   = M_CBRT2 * M_CBRT2;
    double t13  = beta * M_CBRT3 * M_CBRT3;
    double t14  = 1.0 / PI_N1_3;
    double t16  = t13 * t14 * M_CBRT4;
    double t22  = gamma * beta;
    double t46  = t6 * t9;
    double t81  = M_CBRT4 * t9;
    double d2o9 = 2.0 / 9.0;
    
    #pragma omp simd
    for (int i = 0; i < npt; i++)
    {
        double rho_1_3  = cbrt(rho[i]);
        double rho_n1_3 = 1.0 / rho_1_3;
        double rho_n2_3 = rho_n1_3 * rho_n1_3;
        double inv_rho  = rho_n2_3 * rho_n1_3;
        double inv_rho2 = inv_rho  * inv_rho;
        
        double t17 = sigma[i] * t9;
        double t18 = t16 * t17;
        double t21 = inv_rho2 * rho_n2_3;
        double t23 = sqrt(sigma[i]);
        double t24 = t22 * t23;
        double t26 = rho_n2_3 * rho_n2_3;
        double t29 = t23 * M_CBRT2 * t26;
        double t30 = log(t23 * M_CBRT2 * t26 + sqrt(t29 * t29 + 1.0));
        double t31 = M_CBRT2 * t26 * t30;
        double t34 = 1.0 / (1.0 + t24 * t31);
        double t35 = t21 * t34;
        double t39 = 1.0 + d2o9 * t18 * t35;
        double t41 = -0.25 * t7 * t9 * rho_1_3 * t39;
        
        double t45 = -0.1875 * rho_1_3 * rho[i] * M_CBRT3 * PI_N1_3;
        double t49 = rho_n2_3 * inv_rho * inv_rho2;
        double t56 = t35 * t34;
        double t67 = t9 / sqrt(1.0 + t17 * t21);
        double t68 = t22 * t67;
        double t71 = -1.33333333333333 * (M_CBRT2 * rho_n1_3 * inv_rho2 * t24 * t30 + sigma[i] * t49 * t68);
        double t76 = t18 * (-0.59259259259259 * t49 * t34 - d2o9 * t56 * t71);
        
        double t85 = t22 / t23;
        double t91 = 0.5 * (t21 * t68 + t85 * t31);
        double t97 = d2o9 * t46 * (-t18 * t56 * t91 + t13 * t14 * t35 * t81);
        
        exc[i]    = 0.75 * t41;
        vrho[i]   = t41 + t45 * t46 * t76;
        vsigma[i] = t45 * t97;
        
        if (rho[i] < 1e-25)
        {
            exc[i]    = 0.0;
            vrho[i]   = 0.0;
            vsigma[i] = 0.0;
        }
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

// Lee, Yang & Parr correlation
// Based on Libxc implementation, manually optimized
static void eval_GGA_exc_vxc_C_LYP(
    const int npt, const double *rho, const double *sigma, 
    double *exc, double *vrho, double *vsigma
)
{
    const double M_CBRT2 = 1.25992104989487316476721060727;
    const double M_CBRT3 = 1.44224957030740838232163831078;
    const double M_CBRT4 = 1.58740105196819947475170563927;
    const double A       = 0.04918;
    const double B       = 0.132;
    const double c       = 0.2533;
    const double d       = 0.349;
    const double PI_N1_3 = 0.68278406325529568146702083315;  // cbrt(1.0 / M_PI)
    
    double t55    = B * c;
    double t28    = M_PI * M_PI;
    double t29    = cbrt(t28);
    double t40    = M_CBRT3 * t29;
    double t72    = d * d;
    double d1o3   = 1.0 / 3.0;
    double d5o24  = 5.0 / 24.0;
    double d1o18  = 1.0 / 18.0;
    double d1o54  = 1.0 / 54.0;
    double d1o144 = 1.0 / 144.0;
    double t83    = 0.125 - d1o144 * 18.0;
    
    #pragma omp simd
    for (int i = 0; i < npt; i++)
    {
        double rho_1_3  = cbrt(rho[i]);
        double rho_2_3  = rho_1_3 * rho_1_3;
        double rho_n1_3 = 1.0 / rho_1_3;
        double rho_n2_3 = rho_n1_3 * rho_n1_3;
        double inv_rho  = rho_n1_3 * rho_n2_3;
        double rho_n4_3 = rho_n1_3 * inv_rho;
        double inv_rho2 = inv_rho  * inv_rho;
        
        double t11 = 1.0 / (d * rho_n1_3 + 1.0);
        double t13 = exp(-c * rho_n1_3);
        double t14 = B * t13;
        double t18 = rho_n2_3 * inv_rho2;
        double t19 = sigma[i] * t18;
        double t21 = d * t11 + c;
        double t22 = t21 * rho_n1_3;
        double t24 = d1o18 * (-0.25 - 1.75 * t22);
        double t34 = 2.5 - t22 * d1o18;
        double t35 = sigma[i] * t34;
        double t38 = t22 - 11.0;
        double t39 = sigma[i] * t38;
        double t43 = -t19 * (t24 + d5o24) - 0.3 * t40 * t40 
                     +t18 * (0.125 * t35 + d1o144 * t39);
        
        double t47 = rho[i] * A;
        double t49 = t11 * t11;
        double t57 = t13 * t11;
        double t68 = rho_n2_3 * inv_rho * inv_rho2;
        double t69 = 8.0 * sigma[i] * t68;
        double t78 = t21 * rho_n4_3 - t49 * t72 * rho_n2_3 * inv_rho;
        double t79 = 1.75 * d1o54 * t78;
        double t82 = sigma[i] * t78 * d1o54;
        double t95 = d1o3 * (t24 * t69 - t35 * t68) - t19 * t79 
                     + t18 * t82 * t83
                     - d1o54 * t39 * t68 + 1.25 * d1o18 * t69;
        double t97 = d * t49;
        double t98 = rho_n4_3 * d1o3 * (-t97 + t43 * (t55 * t57 + t14 * t97))
                     + t11 * t14 * t95;
        double t107 = t18 * (-t24 + t34 * 0.125 + t38 * d1o144 - d5o24);
        
        exc[i]    = A * t11 * (t14 * t43 - 1.0);
        vrho[i]   = t47 * t98 + exc[i];
        vsigma[i] = B * t47 * t57 * t107;
        
        if (rho[i] < 1e-32)
        {
            exc[i]    = 0.0;
            vrho[i]   = 0.0;
            vsigma[i] = 0.0;
        }
    }
}

// Perdew 86 correlation
// Based on Libxc implementation, manually optimized
static void eval_GGA_exc_vxc_C_P86(
    const int npt, const double *rho, const double *sigma, 
    double *exc, double *vrho, double *vsigma
)
{
    const double M_CBRT3 = 1.44224957030740838232163831078;
    const double M_CBRT4 = 1.58740105196819947475170563927;
    const double PI_N1_3 = 0.68278406325529568146702083315;  // cbrt(1.0 / M_PI)
    
    double t4  = M_CBRT3 * PI_N1_3;
    double t6  = M_CBRT4 * M_CBRT4;
    double t7  = t4 * t6;
    double t32 = M_CBRT3 * M_CBRT3;
    double t33 = PI_N1_3 * PI_N1_3;
    double t34 = M_CBRT4 * t32 * t33;
    double c1  = -0.087741666666666666667 * M_CBRT3 * PI_N1_3;
    
    #pragma omp simd
    for (int i = 0; i < npt; i++)
    {
        double rho_1_3  = cbrt(rho[i]);
        double rho_n1_3 = 1.0 / rho_1_3;
        double rho_n2_3 = rho_n1_3 * rho_n1_3;
        double inv_rho  = rho_n2_3 * rho_n1_3;
        double inv_rho2 = inv_rho  * inv_rho;
        
        double t10 = t7 * rho_n1_3;
        double t11 = t10 * 0.25;
        int    t12 = (1.0 <= t11);
        double t13 = sqrt(t10);
        double t16 = 1.0 / (1.0 + 0.52645 * t13 + 0.08335 * t10);
        double t19 = log(t11);
        double t24 = -0.1423 * t16;
        double t25 = 0.0311 * t19 - 0.048 + t10 * (0.0005 * t19 - 0.0029);
        double t26 = t12 ? t24 : t25;
        double t29 = rho_n1_3 * inv_rho2;
        double t30 = sigma[i] * t29;
        double t38 = t34 * rho_n2_3;
        double t40 = 0.002568 + 0.0058165 * t10 + 0.184725e-5 * t38;
        double t45 = 1.0 / (1.0 + 2.18075 * t10 + 0.118 * t38 + 0.01763993811759021954 * inv_rho);
        double t46 = -t40 * t45;
        double t48 = 0.001667 - t46;
        double t49 = 1.0 / t48;
        double t50 = sqrt(sigma[i]);
        double t51 = t49 * t50;
        double t52 = pow(rho[i], -0.1666666666666667);
        double t54 = t52 * inv_rho;
        double t55 = t51 * t54;
        double t57 = exp(-0.00081290825 * t55);
        double t58 = t57 * t48;
        
        double t66 = rho_n1_3 * inv_rho;
        double t71 = t7 * t66;
        double t73 = c1 / t13 * t6 * t66 - 0.027783333333333333333 * t71;
        double t80 = -t16 * t24 * t73;
        double t81 = -0.010366666666666666667 * inv_rho - t71 * (0.16666666666666666667e-3 * t19 + 0.0008);
        double t82 = t12 ? t80 : t81;
        double t83 = inv_rho2 * inv_rho;
        double t87 = sigma[i] * rho_n1_3 * t58 * t83;
        double t96 = t38 * inv_rho;
        double t98 = -0.001938833333333333333 * t71 - 0.12315e-5 * t96;
        double t107 = -0.72691666666666666667 * t71 - 0.078666666666666666667 * t96 - 0.017639938117590219540 * inv_rho2;
        double t109 = t45 * (t46 * t107 + t98);
        double t117 = 0.00081290825 * t49 * t55 * t109 + 0.94839295833333333334e-3 * inv_rho2 * t51 * t52;
        double t119 = t30 * t57;
        double t120 = t48 * t117 * t119;
        double t122 = t109 * t119;
        double t132 = 0.000406454125 * t50 * t83 / sqrt(rho[i]) * t57;
        
        exc[i]    = t26 + t30 * t58;
        vrho[i]   = exc[i] + rho[i] * (t82 - 7.0/3.0 * t87 + t120 + t122);
        vsigma[i] = rho[i] * (t29 * t57 * t48 - t132);
        
        if (rho[i] < 1e-25)
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
        case GGA_X_B88:  eval_GGA_exc_vxc_X_B88(npt, rho, sigma, exc, vrho, vsigma); break;
        case GGA_C_PBE:  eval_GGA_exc_vxc_C_PBE(npt, rho, sigma, exc, vrho, vsigma); break;
        case GGA_C_LYP:  eval_GGA_exc_vxc_C_LYP(npt, rho, sigma, exc, vrho, vsigma); break;
        case GGA_C_P86:  eval_GGA_exc_vxc_C_P86(npt, rho, sigma, exc, vrho, vsigma); break;
    }
}

