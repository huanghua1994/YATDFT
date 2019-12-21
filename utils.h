#ifndef _YATSCF_UTILS_H_
#define _YATSCF_UTILS_H_

// Helper functions

#define ALIGN64B_MALLOC(x) _mm_malloc((x), 64)
#define ALIGN64B_FREE(x)   _mm_free(x)
#define DBL_SIZE           sizeof(double)
#define INT_SIZE           sizeof(int)

// Copy a block of source matrix to the destination matrix
void copy_matrix_block(
    double *dst, const int ldd, double *src, const int lds, 
    const int nrows, const int ncols
);

// Get current wall-clock time, similar to omp_get_wtime()
double get_wtime_sec();

// For debug, print a dense matrix
void print_mat(double *mat, const int ldm, const int nrows, const int ncols, const char *mat_name);

#endif
