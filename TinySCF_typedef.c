#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <libgen.h>
#include <float.h>
#include <time.h>

#include <mkl.h>

#include "libCMS.h"
#include "utils.h"
#include "TinySCF_typedef.h"
#include "build_HF_mat.h"
#include "build_Dmat.h"
#include "DIIS.h"

void TinySCF_init(TinySCF_t TinySCF, char *bas_fname, char *xyz_fname)
{
    assert(TinySCF != NULL);
    
    double st = get_wtime_sec();
    
    TinySCF->nthread = omp_get_max_threads();
    
    // Reset statistic info
    TinySCF->mem_size       = 0.0;
    TinySCF->init_time      = 0.0;
    TinySCF->S_Hcore_time   = 0.0;
    TinySCF->shell_scr_time = 0.0;
    
    // Load basis set and molecule from input 
    CMS_createBasisSet(&(TinySCF->basis));
    CMS_loadChemicalSystem(TinySCF->basis, bas_fname, xyz_fname);
    int maxAM = CMS_getMaxMomentum(TinySCF->basis);
    TinySCF->bas_name      = basename(bas_fname);
    TinySCF->mol_name      = basename(xyz_fname);
    TinySCF->natom         = CMS_getNumAtoms   (TinySCF->basis);
    TinySCF->nshell        = CMS_getNumShells  (TinySCF->basis);
    TinySCF->nbf           = CMS_getNumFuncs   (TinySCF->basis);
    TinySCF->n_occ         = CMS_getNumOccOrb  (TinySCF->basis);
    TinySCF->charge        = CMS_getTotalCharge(TinySCF->basis);
    TinySCF->electron      = CMS_getNneutral   (TinySCF->basis);
    TinySCF->num_total_sp  = TinySCF->nshell * TinySCF->nshell;
    TinySCF->num_valid_sp  = (TinySCF->nshell + 1) * TinySCF->nshell / 2;
    TinySCF->mat_size      = TinySCF->nbf * TinySCF->nbf;
    TinySCF->max_dim       = (maxAM + 1) * (maxAM + 2) / 2;
    TinySCF->prim_scrtol   = 1e-14;
    TinySCF->shell_scrtol2 = 1e-11 * 1e-11;
    printf("Job information:\n");
    printf("    basis set         = %s\n", TinySCF->bas_name);
    printf("    molecule          = %s\n", TinySCF->mol_name);
    printf("    # atoms           = %d\n", TinySCF->natom);
    printf("    # shells          = %d\n", TinySCF->nshell);
    printf("    # basis functions = %d\n", TinySCF->nbf);
    printf("    # occupied orbits = %d\n", TinySCF->n_occ);
    printf("    # charge          = %d\n", TinySCF->charge);
    printf("    # electrons       = %d\n", TinySCF->electron);
    
    // Allocate memory for ERI info arrays
    CMS_createSimint(TinySCF->basis, &(TinySCF->simint), TinySCF->nthread, TinySCF->prim_scrtol);
    TinySCF->valid_sp_lid  = (int*)    ALIGN64B_MALLOC(INT_SIZE * TinySCF->num_valid_sp);
    TinySCF->valid_sp_rid  = (int*)    ALIGN64B_MALLOC(INT_SIZE * TinySCF->num_valid_sp);
    TinySCF->shell_bf_sind = (int*)    ALIGN64B_MALLOC(INT_SIZE * (TinySCF->nshell + 1));
    TinySCF->shell_bf_num  = (int*)    ALIGN64B_MALLOC(INT_SIZE * TinySCF->nshell);
    TinySCF->sp_scrval     = (double*) ALIGN64B_MALLOC(DBL_SIZE * TinySCF->num_total_sp);
    assert(TinySCF->valid_sp_lid  != NULL);
    assert(TinySCF->valid_sp_rid  != NULL);
    assert(TinySCF->shell_bf_sind != NULL);
    assert(TinySCF->shell_bf_num  != NULL);
    assert(TinySCF->sp_scrval     != NULL);
    TinySCF->mem_size += (double) (INT_SIZE * 2 * TinySCF->num_valid_sp);
    TinySCF->mem_size += (double) (INT_SIZE * (2 * TinySCF->nshell + 1));
    TinySCF->mem_size += (double) (DBL_SIZE * TinySCF->num_total_sp);
    for (int i = 0; i < TinySCF->nshell; i++)
    {
        TinySCF->shell_bf_sind[i] = CMS_getFuncStartInd(TinySCF->basis, i);
        TinySCF->shell_bf_num[i]  = CMS_getShellDim    (TinySCF->basis, i);
    }
    TinySCF->shell_bf_sind[TinySCF->nshell] = TinySCF->nbf;
    
    // Allocate memory for matrices and arrays used only in build_HF_mat
    size_t mat_msize          = DBL_SIZE * TinySCF->mat_size;
    size_t MN_strip_msize     = DBL_SIZE * TinySCF->max_dim * TinySCF->nbf;
    size_t max_buf_entry_size = TinySCF->max_dim * TinySCF->max_dim;
    size_t total_buf_size     = max_buf_entry_size * 6 * TinySCF->nthread;
    TinySCF->max_JKacc_buf    = max_buf_entry_size * 6;
    TinySCF->blk_mat_ptr   = (int*)    ALIGN64B_MALLOC(INT_SIZE * TinySCF->num_total_sp);
    TinySCF->Mpair_flag    = (int*)    ALIGN64B_MALLOC(INT_SIZE * TinySCF->nshell * TinySCF->nthread);
    TinySCF->Npair_flag    = (int*)    ALIGN64B_MALLOC(INT_SIZE * TinySCF->nshell * TinySCF->nthread);
    TinySCF->J_blk_mat     = (double*) ALIGN64B_MALLOC(mat_msize);
    TinySCF->K_blk_mat     = (double*) ALIGN64B_MALLOC(mat_msize);
    TinySCF->D_blk_mat     = (double*) ALIGN64B_MALLOC(mat_msize);
    TinySCF->JKacc_buf     = (double*) ALIGN64B_MALLOC(DBL_SIZE * total_buf_size);
    TinySCF->FM_strip_buf  = (double*) ALIGN64B_MALLOC(MN_strip_msize * TinySCF->nthread);
    TinySCF->FN_strip_buf  = (double*) ALIGN64B_MALLOC(MN_strip_msize * TinySCF->nthread);
    assert(TinySCF->blk_mat_ptr  != NULL);
    assert(TinySCF->Mpair_flag   != NULL);
    assert(TinySCF->Npair_flag   != NULL);
    assert(TinySCF->J_blk_mat    != NULL);
    assert(TinySCF->K_blk_mat    != NULL);
    assert(TinySCF->D_blk_mat    != NULL);
    assert(TinySCF->JKacc_buf    != NULL);
    assert(TinySCF->FM_strip_buf != NULL);
    assert(TinySCF->FN_strip_buf != NULL);
    TinySCF->mem_size += (double) (INT_SIZE * TinySCF->num_total_sp);
    TinySCF->mem_size += (double) (2 * INT_SIZE * TinySCF->nshell * TinySCF->nthread);
    TinySCF->mem_size += (double) (3 * mat_msize);
    TinySCF->mem_size += (double) (2 * MN_strip_msize * TinySCF->nthread);
    TinySCF->mem_size += (double) (DBL_SIZE * total_buf_size);
    int pos = 0, idx = 0;
    for (int i = 0; i < TinySCF->nshell; i++)
    {
        for (int j = 0; j < TinySCF->nshell; j++)
        {
            TinySCF->blk_mat_ptr[idx] = pos;
            pos += TinySCF->shell_bf_num[i] * TinySCF->shell_bf_num[j];
            idx++;
        }
    }
    
    // Allocate memory for matrices used in multiple modules
    TinySCF->tmp_mat = (double*) ALIGN64B_MALLOC(mat_msize);
    assert(TinySCF->tmp_mat != NULL);
    TinySCF->mem_size += (double) (mat_msize);
    
    // Allocate memory for matrices and arrays used only in build_Dmat
    TinySCF->ev_idx = (int*)    ALIGN64B_MALLOC(INT_SIZE * TinySCF->nbf);
    TinySCF->eigval = (double*) ALIGN64B_MALLOC(DBL_SIZE * TinySCF->nbf);
    assert(TinySCF->ev_idx != NULL);
    assert(TinySCF->eigval != NULL);
    TinySCF->mem_size += (double) ((DBL_SIZE + INT_SIZE) * TinySCF->nbf);
    
    // Allocate memory for matrices and arrays used only in DIIS
    size_t DIIS_row_msize = DBL_SIZE * (MAX_DIIS + 1);
    TinySCF->F0_mat    = (double*) ALIGN64B_MALLOC(mat_msize * MAX_DIIS);
    TinySCF->R_mat     = (double*) ALIGN64B_MALLOC(mat_msize * MAX_DIIS);
    TinySCF->B_mat     = (double*) ALIGN64B_MALLOC(DIIS_row_msize * (MAX_DIIS + 1));
    TinySCF->FDS_mat   = (double*) ALIGN64B_MALLOC(mat_msize);
    TinySCF->DIIS_rhs  = (double*) ALIGN64B_MALLOC(DIIS_row_msize);
    TinySCF->DIIS_ipiv = (int*)    ALIGN64B_MALLOC(INT_SIZE * (MAX_DIIS + 1));
    assert(TinySCF->F0_mat    != NULL);
    assert(TinySCF->R_mat     != NULL);
    assert(TinySCF->B_mat     != NULL);
    assert(TinySCF->DIIS_rhs  != NULL);
    assert(TinySCF->DIIS_ipiv != NULL);
    TinySCF->mem_size += MAX_DIIS * 2 * (double) mat_msize;
    TinySCF->mem_size += (double) DIIS_row_msize * (MAX_DIIS + 2);
    TinySCF->mem_size += (double) INT_SIZE * (MAX_DIIS + 1);
    TinySCF->mem_size += (double) mat_msize;
    // Must initialize F0 and R as 0 
    memset(TinySCF->F0_mat, 0, mat_msize * MAX_DIIS);
    memset(TinySCF->R_mat,  0, mat_msize * MAX_DIIS);
    TinySCF->DIIS_len = 0;
    // Initialize B_mat
    for (int i = 0; i < (MAX_DIIS + 1) * (MAX_DIIS + 1); i++)
        TinySCF->B_mat[i] = -1.0;
    for (int i = 0; i < (MAX_DIIS + 1); i++)
        TinySCF->B_mat[i * (MAX_DIIS + 1) + i] = 0.0;
    TinySCF->DIIS_bmax_id = 0;
    TinySCF->DIIS_bmax    = -DBL_MAX;

    // Allocate memory for matrices and arrays used only in SCF iterations
    TinySCF->ene_tol  = 1e-11;
    TinySCF->Hcore_mat  = (double*) ALIGN64B_MALLOC(mat_msize);
    TinySCF->S_mat      = (double*) ALIGN64B_MALLOC(mat_msize);
    TinySCF->F_mat      = (double*) ALIGN64B_MALLOC(mat_msize);
    TinySCF->D_mat      = (double*) ALIGN64B_MALLOC(mat_msize);
    TinySCF->J_mat      = (double*) ALIGN64B_MALLOC(mat_msize);
    TinySCF->K_mat      = (double*) ALIGN64B_MALLOC(mat_msize);
    TinySCF->X_mat      = (double*) ALIGN64B_MALLOC(mat_msize);
    TinySCF->Cocc_mat   = (double*) ALIGN64B_MALLOC(DBL_SIZE * TinySCF->n_occ * TinySCF->nbf);
    assert(TinySCF->Hcore_mat != NULL);
    assert(TinySCF->S_mat     != NULL);
    assert(TinySCF->F_mat     != NULL);
    assert(TinySCF->D_mat     != NULL);
    assert(TinySCF->J_mat     != NULL);
    assert(TinySCF->K_mat     != NULL);
    assert(TinySCF->X_mat     != NULL);
    assert(TinySCF->Cocc_mat  != NULL);
    TinySCF->mem_size += (double) (7 * mat_msize);
    TinySCF->mem_size += (double) (DBL_SIZE * TinySCF->n_occ * TinySCF->nbf);

    double et = get_wtime_sec();
    TinySCF->init_time = et - st;
    
    // Print memory usage and time consumption
    printf("TinySCF memory usage    = %.2lf MB\n", TinySCF->mem_size / 1048576.0);
    printf("TinySCF memory allocation and initialization over, elapsed time = %.3lf (s)\n", TinySCF->init_time);
}

void TinySCF_destroy(TinySCF_t TinySCF)
{
    assert(TinySCF != NULL);
    
    // Free Fock accumulation buffer
    ALIGN64B_FREE(TinySCF->JKacc_buf);
    
    // Free ERI info arrays
    ALIGN64B_FREE(TinySCF->valid_sp_lid);
    ALIGN64B_FREE(TinySCF->valid_sp_rid);
    ALIGN64B_FREE(TinySCF->shell_bf_sind);
    ALIGN64B_FREE(TinySCF->shell_bf_num);
    ALIGN64B_FREE(TinySCF->sp_scrval);
    
    // Free matrices and temporary arrays used only in build_HF_mat
    ALIGN64B_FREE(TinySCF->blk_mat_ptr);
    ALIGN64B_FREE(TinySCF->Mpair_flag);
    ALIGN64B_FREE(TinySCF->Npair_flag);
    ALIGN64B_FREE(TinySCF->J_blk_mat);
    ALIGN64B_FREE(TinySCF->K_blk_mat);
    ALIGN64B_FREE(TinySCF->D_blk_mat);
    ALIGN64B_FREE(TinySCF->JKacc_buf);
    ALIGN64B_FREE(TinySCF->FM_strip_buf);
    ALIGN64B_FREE(TinySCF->FN_strip_buf);
    
    // Free matrices used in multiple modules
    ALIGN64B_FREE(TinySCF->tmp_mat);
    
    // Free matrices and arrays used only in build_Dmat
    ALIGN64B_FREE(TinySCF->ev_idx);
    ALIGN64B_FREE(TinySCF->eigval);
    
    // Free matrices and temporary arrays used only in DIIS
    ALIGN64B_FREE(TinySCF->F0_mat);
    ALIGN64B_FREE(TinySCF->R_mat);
    ALIGN64B_FREE(TinySCF->B_mat);
    ALIGN64B_FREE(TinySCF->FDS_mat);
    ALIGN64B_FREE(TinySCF->DIIS_rhs);
    ALIGN64B_FREE(TinySCF->DIIS_ipiv);
    
    // Free matrices and temporary arrays used only in SCF
    ALIGN64B_FREE(TinySCF->Hcore_mat);
    ALIGN64B_FREE(TinySCF->S_mat);
    ALIGN64B_FREE(TinySCF->F_mat);
    ALIGN64B_FREE(TinySCF->D_mat);
    ALIGN64B_FREE(TinySCF->J_mat);
    ALIGN64B_FREE(TinySCF->K_mat);
    ALIGN64B_FREE(TinySCF->X_mat);
    ALIGN64B_FREE(TinySCF->Cocc_mat);

    // Free BasisSet_t and Simint_t object, print Simint_t object stat info
    CMS_destroyBasisSet(TinySCF->basis);
    CMS_destroySimint(TinySCF->simint, 1);
    
    free(TinySCF);
}

static int cmp_pair(int M1, int N1, int M2, int N2)
{
    if (M1 == M2) return (N1 < N2);
    else return (M1 < M2);
}

static void quickSort(int *M, int *N, int l, int r)
{
    int i = l, j = r, tmp;
    int mid_M = M[(i + j) / 2];
    int mid_N = N[(i + j) / 2];
    while (i <= j)
    {
        while (cmp_pair(M[i], N[i], mid_M, mid_N)) i++;
        while (cmp_pair(mid_M, mid_N, M[j], N[j])) j--;
        if (i <= j)
        {
            tmp = M[i]; M[i] = M[j]; M[j] = tmp;
            tmp = N[i]; N[i] = N[j]; N[j] = tmp;
            
            i++;  j--;
        }
    }
    if (i < r) quickSort(M, N, i, r);
    if (j > l) quickSort(M, N, l, j);
}

void TinySCF_screen_shell_quartets(TinySCF_t TinySCF)
{
    assert(TinySCF != NULL);
    
    int    nshell        = TinySCF->nshell;
    int    *shell_bf_num = TinySCF->shell_bf_num;
    int    *valid_sp_lid = TinySCF->valid_sp_lid;
    int    *valid_sp_rid = TinySCF->valid_sp_rid;
    double shell_scrtol2 = TinySCF->shell_scrtol2;
    double *sp_scrval    = TinySCF->sp_scrval;
    Simint_t simint      = TinySCF->simint;
    
    double st = get_wtime_sec();
    
    // Compute screening values using Schwarz inequality
    double global_max_scrval = 0.0;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic) reduction(max:global_max_scrval)
        for (int M = 0; M < nshell; M++)
        {
            int dimM = shell_bf_num[M];
            for (int N = 0; N < nshell; N++)
            {
                int dimN = shell_bf_num[N];
                
                int nints;
                double *integrals;
                CMS_computeShellQuartet_Simint(simint, tid, M, N, M, N, &integrals, &nints);
                
                double maxval = 0.0;
                if (nints > 0)
                {
                    // Loop over all ERIs in a shell quartet and find the max value
                    for (int iM = 0; iM < dimM; iM++)
                        for (int iN = 0; iN < dimN; iN++)
                        {
                            int index = iN * (dimM * dimN * dimM + dimM) + iM * (dimN * dimM + 1); // Simint layout
                            double val = fabs(integrals[index]);
                            if (val > maxval) maxval = val;
                        }
                }
                sp_scrval[M * nshell + N] = maxval;
                if (maxval > global_max_scrval) global_max_scrval = maxval;
            }
        }
    }
    
    // Reset Simint statistic info
    CMS_Simint_resetStatisInfo(simint);
    
    // Generate unique shell pairs that survive Schwarz screening
    // eta is the threshold for screening a shell pair
    double eta = shell_scrtol2 / global_max_scrval;
    int num_valid_sp = 0;
    for (int M = 0; M < nshell; M++)
    {
        for (int N = 0; N < nshell; N++)
        {
            double MN_scrval = sp_scrval[M * nshell + N];
            // if sp_scrval * max_scrval < shell_scrtol2, for any given shell pair
            // (P,Q), (MN|PQ) is always < shell_scrtol2 and will be screened
            if (MN_scrval > eta)  
            {
                // Make {N_i} in (M, N_i) as continuous as possible to get better
                // memory access pattern and better performance
                if (N > M) continue;
                
                // We want AM(M) >= AM(N) to avoid HRR
                int MN_id = CMS_Simint_getShellpairAMIndex(simint, M, N);
                int NM_id = CMS_Simint_getShellpairAMIndex(simint, N, M);
                if (MN_id > NM_id)
                {
                    valid_sp_lid[num_valid_sp] = M;
                    valid_sp_rid[num_valid_sp] = N;
                } else {
                    valid_sp_lid[num_valid_sp] = N;
                    valid_sp_rid[num_valid_sp] = M;
                }
                num_valid_sp++;
            }
        }
    }
    TinySCF->num_valid_sp = num_valid_sp;
    quickSort(valid_sp_lid, valid_sp_rid, 0, num_valid_sp - 1);
    
    double et = get_wtime_sec();
    TinySCF->shell_scr_time = et - st;
    
    // Print runtime
    printf("TinySCF precompute shell screening info over,      elapsed time = %.3lf (s)\n", TinySCF->shell_scr_time);
}

// Compute Hartree-Fock energy
static void TinySCF_calc_energy(TinySCF_t TinySCF)
{
    double energy = 0.0;
    
    #pragma omp simd 
    for (int i = 0; i < TinySCF->mat_size; i++)
        energy += TinySCF->D_mat[i] * (TinySCF->F_mat[i] + TinySCF->Hcore_mat[i]);
    
    TinySCF->HF_energy = energy;
}

void TinySCF_do_SCF(TinySCF_t TinySCF, const int max_iter)
{
    // Start SCF iterations
    printf("TinySCF SCF iteration started...\n");
    printf("Nuclear energy = %.10lf\n", TinySCF->nuc_energy);
    TinySCF->iter = 0;
    TinySCF->max_iter = max_iter;
    double prev_energy  = 0;
    double energy_delta = 223;
    
    int    nbf        = TinySCF->nbf;
    int    mat_size   = TinySCF->mat_size;
    double *D_mat     = TinySCF->D_mat;
    double *J_mat     = TinySCF->J_mat;
    double *K_mat     = TinySCF->K_mat;
    double *F_mat     = TinySCF->F_mat;
    double *X_mat     = TinySCF->X_mat;
    double *S_mat     = TinySCF->S_mat;
    double *Hcore_mat = TinySCF->Hcore_mat;
    double *Cocc_mat  = TinySCF->Cocc_mat;

    while ((TinySCF->iter < TinySCF->max_iter) && (energy_delta >= TinySCF->ene_tol))
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
        TinySCF_calc_energy(TinySCF);
        et1 = get_wtime_sec();
        printf("* Calculate energy      : %.3lf (s)\n", et1 - st1);
        energy_delta = fabs(TinySCF->HF_energy - prev_energy);
        prev_energy = TinySCF->HF_energy;
        
        // DIIS acceleration (Pulay mixing)
        st1 = get_wtime_sec();
        TinySCF_DIIS(TinySCF, X_mat, S_mat, D_mat, F_mat);
        et1 = get_wtime_sec();
        printf("* DIIS procedure        : %.3lf (s)\n", et1 - st1);
        
        // Diagonalize and build the density matrix
        st1 = get_wtime_sec();
        TinySCF_build_Dmat_eig(TinySCF, F_mat, X_mat, D_mat, Cocc_mat);
        et1 = get_wtime_sec(); 
        printf("* Build density matrix  : %.3lf (s)\n", et1 - st1);
        
        et0 = get_wtime_sec();
        
        printf("* Iteration runtime     = %.3lf (s)\n", et0 - st0);
        printf("* Energy = %.10lf (%.10lf)", TinySCF->HF_energy + TinySCF->nuc_energy, TinySCF->HF_energy);
        if (TinySCF->iter > 0) 
        {
            printf(", delta = %e\n", energy_delta); 
        } else 
        {
            printf("\n");
            energy_delta = 223;  // Prevent the SCF exit after 1st iteration when no SAD initial guess
        }
        
        TinySCF->iter++;
    }
    printf("--------------- SCF iterations finished ---------------\n");
}

