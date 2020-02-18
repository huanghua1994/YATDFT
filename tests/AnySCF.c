#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "TinyDFT.h"

void save_mat_to_file(
    const char *env_str, const char *format_str, const int nbf, 
    const char *mol_name, const char *bas_name, const double *mat
)
{
    if (env_str == NULL) return;
    int need_save = atoi(env_str);
    if (need_save != 1) return;
    
    char ouf_name[256];
    sprintf(ouf_name, format_str, nbf, mol_name, bas_name);
    FILE *ouf = fopen(ouf_name, "wb");
    fwrite(mat, sizeof(double), nbf * nbf, ouf);
    fclose(ouf);
    printf("Binary file %s output finished\n", ouf_name);
}

void TinyDFT_SCF(TinyDFT_t TinyDFT, const int max_iter, int J_op, int K_op)
{
    // Start SCF iterations
    printf("Self-Consistent Field iteration started...\n");
    printf("Nuclear repulsion energy = %.10lf\n", TinyDFT->E_nuc_rep);
    TinyDFT->iter = 0;
    TinyDFT->max_iter = max_iter;
    double E_prev, E_curr, E_delta = 19241112.0;
    
    int    nbf            = TinyDFT->nbf;
    int    mat_size       = TinyDFT->mat_size;
    int    xf_id          = TinyDFT->xf_id;
    int    xf_family      = TinyDFT->xf_family;
    double *Hcore_mat     = TinyDFT->Hcore_mat;
    double *S_mat         = TinyDFT->S_mat;
    double *X_mat         = TinyDFT->X_mat;
    double *J_mat         = TinyDFT->J_mat;
    double *K_mat         = TinyDFT->K_mat;
    double *XC_mat        = TinyDFT->XC_mat;
    double *F_mat         = TinyDFT->F_mat;
    double *Cocc_mat      = TinyDFT->Cocc_mat;
    double *D_mat         = TinyDFT->D_mat;
    double *E_nuc_rep     = &TinyDFT->E_nuc_rep;
    double *E_one_elec    = &TinyDFT->E_one_elec;
    double *E_two_elec    = &TinyDFT->E_two_elec;
    double *E_HF_exchange = &TinyDFT->E_HF_exchange;
    double *E_DFT_XC      = &TinyDFT->E_DFT_XC;

    int J_direct = 0, K_direct = 0, JK_direct = 0;
    int J_denfit = 0, K_denfit = 0, K_xc = 0, xc_hybrid = 0;
    if (xf_family == FAMILY_HYB_GGA) xc_hybrid = 1;
    if (J_op == 0) J_direct = 1;
    if (J_op == 1) J_denfit = 1;
    if (K_op == 0) K_direct = 1;
    if (K_op == 1) K_denfit = 1;
    if (K_op == 2) K_xc     = 1;
    if (xc_hybrid == 1)
    {
        if (J_direct == 1) K_direct = 1;
        if (J_denfit == 1) K_denfit = 1;
    }
    JK_direct = J_direct & K_direct;
    double HF_x_coef;
    if (xf_id == HYB_GGA_XC_B3LYP || xf_id == HYB_GGA_XC_B3LYP5) HF_x_coef = 0.2;

    while ((TinyDFT->iter < TinyDFT->max_iter) && (fabs(E_delta) >= TinyDFT->E_tol))
    {
        printf("--------------- Iteration %d ---------------\n", TinyDFT->iter);
        
        double st0, et0, st1, et1, st2;
        double J_time = 0, K_time = 0, XC_time = 0;
        st0 = get_wtime_sec();
        
        // Build the Fock matrix
        if (JK_direct == 1)
        {
            st1 = get_wtime_sec();
            TinyDFT_build_JKmat(TinyDFT, D_mat, J_mat, K_mat);
            st2 = get_wtime_sec();
            J_time = 0.5 * (st2 - st1);
            K_time = 0.5 * (st2 - st1);
        }
        if (JK_direct == 0 && J_direct == 1)
        {
            st1 = get_wtime_sec();
            TinyDFT_build_JKmat(TinyDFT, D_mat, J_mat, NULL);
            st2 = get_wtime_sec();
            J_time = st2 - st1;
        }
        if (J_denfit == 1)
        {
            st1 = get_wtime_sec();
            TinyDFT_build_JKmat_DF(TinyDFT, D_mat, Cocc_mat, J_mat, NULL);
            st2 = get_wtime_sec();
            J_time = st2 - st1;
        }
        if (JK_direct == 0 && K_direct == 1)
        {
            st1 = get_wtime_sec();
            TinyDFT_build_JKmat(TinyDFT, D_mat, NULL, K_mat);
            st2 = get_wtime_sec();
            K_time = st2 - st1;
        }
        if (K_denfit == 1)
        {
            st1 = get_wtime_sec();
            TinyDFT_build_JKmat_DF(TinyDFT, D_mat, Cocc_mat, NULL, K_mat);
            st2 = get_wtime_sec();
            K_time = st2 - st1;
        }
        if (K_xc == 1)
        {
            st1 = get_wtime_sec();
            *E_DFT_XC = TinyDFT_build_XC_mat(TinyDFT, D_mat, XC_mat);
            st2 = get_wtime_sec();
            XC_time = st2 - st1;
        }
        if (K_op == 0 || K_op == 1)
        {
            #pragma omp parallel for simd
            for (int i = 0; i < mat_size; i++)
                F_mat[i] = Hcore_mat[i] + 2 * J_mat[i] - K_mat[i];
        }
        if (K_op == 2 && xc_hybrid == 0)
        {
            #pragma omp parallel for simd
            for (int i = 0; i < mat_size; i++)
                F_mat[i] = Hcore_mat[i] + 2 * J_mat[i] + XC_mat[i];
        }
        if (K_op == 2 && xc_hybrid == 1)
        {
            #pragma omp parallel for simd
            for (int i = 0; i < mat_size; i++)
                F_mat[i] = Hcore_mat[i] + 2 * J_mat[i] + XC_mat[i] - HF_x_coef * K_mat[i];
        }
        et1 = get_wtime_sec();
        printf(
            "* Build Fock matrix     : %.3lf (s), J, K, XC = %.3lf, %.3lf, %.3lf (s)\n", 
            et1 - st0, J_time, K_time, XC_time
        );
        
        // Calculate new system energy
        st1 = get_wtime_sec();
        if (K_direct == 1 || K_denfit == 1)
        {
            TinyDFT_calc_HF_energy(
                mat_size, D_mat, Hcore_mat, J_mat, K_mat, 
                E_one_elec, E_two_elec, E_HF_exchange
            );
        } else {
            TinyDFT_calc_HF_energy(
                mat_size, D_mat, Hcore_mat, J_mat, NULL, 
                E_one_elec, E_two_elec, NULL
            );
        }
        E_curr = (*E_nuc_rep) + (*E_one_elec) + (*E_two_elec);
        if (K_op == 0 || K_op == 1) E_curr += (*E_HF_exchange);
        if (K_op == 2) E_curr += (*E_DFT_XC);
        if (K_op == 2 && xc_hybrid == 1) E_curr += HF_x_coef * (*E_HF_exchange);
        et1 = get_wtime_sec();
        printf("* Calculate energy      : %.3lf (s)\n", et1 - st1);
        E_delta = E_curr - E_prev;
        E_prev = E_curr;
        
        if (TinyDFT->iter == max_iter - 1)
        {
            char *mol_name = TinyDFT->mol_name;
            char *bas_name = TinyDFT->bas_name;
            save_mat_to_file(getenv("OUTPUT_D"),  "D_%d_%s_%s.bin",  nbf, mol_name, bas_name, D_mat);
            save_mat_to_file(getenv("OUTPUT_J"),  "J_%d_%s_%s.bin",  nbf, mol_name, bas_name, J_mat);
            save_mat_to_file(getenv("OUTPUT_K"),  "K_%d_%s_%s.bin",  nbf, mol_name, bas_name, K_mat);
            save_mat_to_file(getenv("OUTPUT_XC"), "XC_%d_%s_%s.bin", nbf, mol_name, bas_name, XC_mat);
        }
        
        // CDIIS acceleration (Pulay mixing)
        st1 = get_wtime_sec();
        TinyDFT_CDIIS(TinyDFT, X_mat, S_mat, D_mat, F_mat);
        et1 = get_wtime_sec();
        printf("* CDIIS procedure       : %.3lf (s)\n", et1 - st1);
        
        // Diagonalize and build the density matrix
        st1 = get_wtime_sec();
        TinyDFT_build_Dmat_eig(TinyDFT, F_mat, X_mat, D_mat, Cocc_mat);
        et1 = get_wtime_sec(); 
        printf("* Build density matrix  : %.3lf (s)\n", et1 - st1);
        
        et0 = get_wtime_sec();
        
        printf("* Iteration runtime     = %.3lf (s)\n", et0 - st0);
        printf("* Energy = %.10lf", E_curr);
        if (TinyDFT->iter > 0) 
        {
            printf(", delta = %e\n", E_delta); 
        } else {
            printf("\n");
            E_delta = 19241112.0;  // Prevent the SCF exit after 1st iteration when no SAD initial guess
        }
        
        TinyDFT->iter++;
        fflush(stdout);
    }
    printf("--------------- SCF iterations finished ---------------\n");
}

void print_usage(const char *argv0)
{
    printf("Usage: %s <basis> <xyz> <niter> <direct/DF J> <direct/DF/DFT K/XC> <df_basis> <X-func> <C-func>\n", argv0);
    printf("  * direct/DF J: 0 for direct method, 1 for density fitting\n");
    printf("  * direct/DF/DFT K/XC: 0 for direct method K, 1 for density fitting K, 2 for DFT XC\n");
    printf("  * available XC functions: LDA_X, LDA_C_XA, LDA_C_PZ, LDA_C_PW,\n");
    printf("                            GGA_X_PBE, GGA_X_B88, GGA_C_PBE, GGA_C_LYP, \n");
    printf("                            HYB_GGA_XC_B3LYP, HYB_GGA_XC_B3LYP5\n");
    printf("  Note: if you use hybrid GGA functionals, enter it twice for both <X-func> and <C-func>.\n");
}

int main(int argc, char **argv)
{
    if (argc < 6)
    {
        print_usage(argv[0]);
        return 255;
    }
    
    double st, et;
    
    int niter, J_op, K_op, use_DF = 0;
    
    niter = atoi(argv[3]);
    J_op  = atoi(argv[4]);
    K_op  = atoi(argv[5]);
    if (J_op < 0 || J_op > 1) J_op = 0;
    if (K_op < 0 || K_op > 2) K_op = 0;
    printf("%s will use: ", argv[0]);
    if (J_op == 0) printf("direct J, ");
    if (J_op == 1) printf("denfit J, ");
    if (K_op == 0) printf("direct K\n");
    if (K_op == 1) printf("denfit K\n");
    if (K_op == 2) printf("DFT XC\n");
    
    // Initialize TinyDFT
    TinyDFT_t TinyDFT;
    TinyDFT_init(&TinyDFT, argv[1], argv[2]);
    
    // Compute constant matrices and get initial guess for D
    st = get_wtime_sec();
    TinyDFT_build_Hcore_S_X_mat(TinyDFT, TinyDFT->Hcore_mat, TinyDFT->S_mat, TinyDFT->X_mat);
    TinyDFT_build_Dmat_SAD(TinyDFT, TinyDFT->D_mat);
    et = get_wtime_sec();
    printf("TinyDFT compute Hcore, S, X matrices over, elapsed time = %.3lf (s)\n", et - st);
    
    // Set up density fitting
    if (J_op == 1 || K_op == 1)
    {
        if (argc < 7)
        {
            printf("You need to provide a density fitting auxiliary basis set!\n");
            print_usage(argv[0]);
            return 255;
        }
        use_DF = 1;
        // If we don't need DF for K build, reduce memory usage in DF, only DF tensor build
        // will become slower; otherwise, use more memory in DF for better K build performance
        if (K_op == 1)
        {
            TinyDFT_setup_DF(TinyDFT, argv[6], argv[2], 0);
        } else {
            TinyDFT_setup_DF(TinyDFT, argv[6], argv[2], 1);
        }
        TinyDFT_build_Cocc_from_Dmat(TinyDFT, TinyDFT->D_mat, TinyDFT->Cocc_mat);
    }
    
    // Set up XC numerical integral environments
    if (K_op == 2)
    {
        char default_xf_str[6]  = "LDA_X\0";
        char default_cf_str[10] = "LDA_C_PW\0";
        char *xf_str = &default_xf_str[0];
        char *cf_str = &default_cf_str[0];
        if (use_DF == 1 && argc >= 9)
        {
            xf_str = argv[7];
            cf_str = argv[8];
        }
        if (use_DF == 0 && argc >= 8)
        {
            xf_str = argv[6];
            cf_str = argv[7];
        }
        st = get_wtime_sec();
        TinyDFT_setup_XC_integral(TinyDFT, xf_str, cf_str);
        et = get_wtime_sec();
        printf("TinyDFT set up XC integral over, elapsed time = %.3lf (s)\n", et - st);
    }
    
    // Do SCF calculation
    TinyDFT_SCF(TinyDFT, niter, J_op, K_op);
    
    // Free TinyDFT and H2P-ERI
    TinyDFT_destroy(&TinyDFT);
    
    return 0;
}
