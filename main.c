#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <assert.h>

#include "TinySCF_typedef.h"
#include "build_Dmat.h"
#include "build_HF_mat.h"

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s <basis> <xyz> <niter>\n", argv[0]);
        return 255;
    }
    
    // Initialize TinySCF
    TinySCF_t TinySCF = (TinySCF_t) malloc(sizeof(struct TinySCF_struct));
    TinySCF_init(TinySCF, argv[1], argv[2]);
    TinySCF_screen_shell_quartets(TinySCF);
    
    // Precompute constant matrices and get initial guess for D
    TinySCF_build_Hcore_S_X_mat(TinySCF, TinySCF->Hcore_mat, TinySCF->S_mat, TinySCF->X_mat);
    TinySCF_build_Dmat_SAD(TinySCF, TinySCF->D_mat);
    TinySCF->nuc_energy = CMS_getNucEnergy(TinySCF->basis);
    
    // Do SCF calculation
    TinySCF_do_SCF(TinySCF, atoi(argv[3]));
    
    // Free TinySCF and H2P-ERI
    TinySCF_destroy(TinySCF);
    
    return 0;
}
