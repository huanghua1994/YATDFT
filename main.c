#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "TinySCF_typedef.h"
#include "build_Dmat.h"
#include "build_HF_mat.h"
#include "CDIIS.h"
#include "utils.h"

void TinySCF_HFSCF(TinySCF_t TinySCF, const int max_iter)
{
    // Start SCF iterations
    printf("TinySCF SCF iteration started...\n");
    printf("Nuclear repulsion energy = %.10lf\n", TinySCF->E_nuc_rep);
    TinySCF->iter = 0;
    TinySCF->max_iter = max_iter;
    double E_prev, E_curr, E_delta = 19241112.0;
    
    int    nbf            = TinySCF->nbf;
    int    mat_size       = TinySCF->mat_size;
    double *D_mat         = TinySCF->D_mat;
    double *J_mat         = TinySCF->J_mat;
    double *K_mat         = TinySCF->K_mat;
    double *F_mat         = TinySCF->F_mat;
    double *X_mat         = TinySCF->X_mat;
    double *S_mat         = TinySCF->S_mat;
    double *Hcore_mat     = TinySCF->Hcore_mat;
    double *Cocc_mat      = TinySCF->Cocc_mat;
    double *E_nuc_rep     = &TinySCF->E_nuc_rep;
    double *E_one_elec    = &TinySCF->E_one_elec;
    double *E_two_elec    = &TinySCF->E_two_elec;
    double *E_HF_exchange = &TinySCF->E_HF_exchange;

    while ((TinySCF->iter < TinySCF->max_iter) && (fabs(E_delta) >= TinySCF->E_tol))
    {
        printf("--------------- Iteration %d ---------------\n", TinySCF->iter);
        
        double st0, et0, st1, et1, st2;
        st0 = get_wtime_sec();
        
        // Build the Fock matrix
        st1 = get_wtime_sec();
        TinySCF_build_JKmat(TinySCF, D_mat, J_mat, K_mat);
        #pragma omp for simd
        for (int i = 0; i < mat_size; i++)
            F_mat[i] = Hcore_mat[i] + 2 * J_mat[i] - K_mat[i];
        et1 = get_wtime_sec();
        printf("* Build Fock matrix     : %.3lf (s)\n", et1 - st1);
        
        // Calculate new system energy
        st1 = get_wtime_sec();
        TinySCF_calc_HF_energy(
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
        TinySCF_CDIIS(TinySCF, X_mat, S_mat, D_mat, F_mat);
        et1 = get_wtime_sec();
        printf("* CDIIS procedure       : %.3lf (s)\n", et1 - st1);
        
        // Diagonalize and build the density matrix
        st1 = get_wtime_sec();
        TinySCF_build_Dmat_eig(TinySCF, F_mat, X_mat, D_mat, Cocc_mat);
        et1 = get_wtime_sec(); 
        printf("* Build density matrix  : %.3lf (s)\n", et1 - st1);
        
        et0 = get_wtime_sec();
        
        printf("* Iteration runtime     = %.3lf (s)\n", et0 - st0);
        printf("* Energy = %.10lf", E_curr);
        if (TinySCF->iter > 0) 
        {
            printf(", delta = %e\n", E_delta); 
        } else {
            printf("\n");
            E_delta = 19241112.0;  // Prevent the SCF exit after 1st iteration when no SAD initial guess
        }
        
        TinySCF->iter++;
    }
    printf("--------------- SCF iterations finished ---------------\n");
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s <basis> <xyz> <niter>\n", argv[0]);
        return 255;
    }
    
    // Initialize TinySCF
    TinySCF_t TinySCF;
    TinySCF_init(&TinySCF, argv[1], argv[2]);
    
    // Compute constant matrices and get initial guess for D
    TinySCF_build_Hcore_S_X_mat(TinySCF, TinySCF->Hcore_mat, TinySCF->S_mat, TinySCF->X_mat);
    TinySCF_build_Dmat_SAD(TinySCF, TinySCF->D_mat);
    
    // Do HFSCF calculation
    TinySCF_HFSCF(TinySCF, atoi(argv[3]));
    
    // Free TinySCF and H2P-ERI
    TinySCF_destroy(&TinySCF);
    
    return 0;
}
