#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "TinyDFT.h"

void TinyDFT_HFSCF(TinyDFT_t TinyDFT, const int max_iter)
{
    // Start SCF iterations
    printf("HFSCF iteration started...\n");
    printf("Nuclear repulsion energy = %.10lf\n", TinyDFT->E_nuc_rep);
    TinyDFT->iter = 0;
    TinyDFT->max_iter = max_iter;
    double E_prev, E_curr, E_delta = 19241112.0;
    
    int    nbf            = TinyDFT->nbf;
    int    mat_size       = TinyDFT->mat_size;
    double *D_mat         = TinyDFT->D_mat;
    double *J_mat         = TinyDFT->J_mat;
    double *K_mat         = TinyDFT->K_mat;
    double *F_mat         = TinyDFT->F_mat;
    double *X_mat         = TinyDFT->X_mat;
    double *S_mat         = TinyDFT->S_mat;
    double *Hcore_mat     = TinyDFT->Hcore_mat;
    double *Cocc_mat      = TinyDFT->Cocc_mat;
    double *E_nuc_rep     = &TinyDFT->E_nuc_rep;
    double *E_one_elec    = &TinyDFT->E_one_elec;
    double *E_two_elec    = &TinyDFT->E_two_elec;
    double *E_HF_exchange = &TinyDFT->E_HF_exchange;

    while ((TinyDFT->iter < TinyDFT->max_iter) && (fabs(E_delta) >= TinyDFT->E_tol))
    {
        printf("--------------- Iteration %d ---------------\n", TinyDFT->iter);
        
        double st0, et0, st1, et1, st2;
        st0 = get_wtime_sec();
        
        // Build the Fock matrix
        st1 = get_wtime_sec();
        TinyDFT_build_JKmat_DF(TinyDFT, D_mat, Cocc_mat, J_mat, K_mat);
        #pragma omp parallel for simd
        for (int i = 0; i < mat_size; i++)
            F_mat[i] = Hcore_mat[i] + 2 * J_mat[i] - K_mat[i];
        et1 = get_wtime_sec();
        printf("* Build Fock matrix     : %.3lf (s)\n", et1 - st1);
        
        // Calculate new system energy
        st1 = get_wtime_sec();
        TinyDFT_calc_HF_energy(
            mat_size, D_mat, Hcore_mat, J_mat, K_mat, 
            E_one_elec, E_two_elec, E_HF_exchange
        );
        E_curr = (*E_nuc_rep) + (*E_one_elec) + (*E_two_elec) + (*E_HF_exchange);
        et1 = get_wtime_sec();
        printf("* Calculate energy      : %.3lf (s)\n", et1 - st1);
        E_delta = E_curr - E_prev;
        E_prev = E_curr;
        
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
    }
    printf("--------------- SCF iterations finished ---------------\n");
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s <basis> <df_basis> <xyz> <niter>\n", argv[0]);
        return 255;
    }
    
    // Initialize TinyDFT
    TinyDFT_t TinyDFT;
    TinyDFT_init(&TinyDFT, argv[1], argv[3]);
    
    // Compute constant matrices and get initial guess for D
    TinyDFT_build_Hcore_S_X_mat(TinyDFT, TinyDFT->Hcore_mat, TinyDFT->S_mat, TinyDFT->X_mat);
    //TinyDFT_build_Dmat_SAD(TinyDFT, TinyDFT->D_mat);
    memset(TinyDFT->D_mat, 0, sizeof(double) * TinyDFT->nbf * TinyDFT->nbf);
    
    // Set up density fitting
    TinyDFT_setup_DF(TinyDFT, argv[2], argv[3]);
    
    // Do HFSCF calculation
    TinyDFT_HFSCF(TinyDFT, atoi(argv[4]));
    
    // Free TinyDFT and H2P-ERI
    TinyDFT_destroy(&TinyDFT);
    
    return 0;
}
