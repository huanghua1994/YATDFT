#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <assert.h>

#include "TinySCF.h"
#include "utils.h"

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s <basis> <xyz> <niter>\n", argv[0]);
        return 255;
    }
    
    // Initialize TinySCF
    TinySCF_t TinySCF = (TinySCF_t) malloc(sizeof(struct TinySCF_struct));
    TinySCF_init(TinySCF, argv[1], argv[2], atoi(argv[3]));
    TinySCF_compute_Hcore_Ovlp_mat(TinySCF);
    TinySCF_compute_sq_Schwarz_scrvals(TinySCF);
    TinySCF_get_initial_guess(TinySCF);
    
    // Do SCF calculation
    TinySCF_do_SCF(TinySCF);
    
    // Free TinySCF and H2P-ERI
    TinySCF_destroy(TinySCF);
    
    return 0;
}
