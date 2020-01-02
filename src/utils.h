#ifndef __UTILS_H__
#define __UTILS_H__

// Helper functions

#define ALIGN64B_MALLOC(x) _mm_malloc((x), 64)
#define ALIGN64B_FREE(x)   _mm_free(x)
#define DBL_SIZE           sizeof(double)
#define INT_SIZE           sizeof(int)

#ifdef __cplusplus
extern "C" {
#endif

// Copy a block of source matrix to the destination matrix
void copy_dbl_mat_blk(
    double *dst, const int ldd, const double *src, const int lds, 
    const int nrows, const int ncols
);

// Get current wall-clock time, similar to omp_get_wtime()
double get_wtime_sec();

// A safe version of (iblk * len) / nblk
int block_spos(const int nblk, const int iblk, const int len);

// For debug, print a dense matrix
void print_dbl_mat(double *mat, const int ldm, const int nrows, const int ncols, const char *mat_name);

#ifdef __cplusplus
}
#endif

#endif
