#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#include "utils.h"

void copy_dbl_mat_blk(
    double *dst, const int ldd, const double *src, const int lds, 
    const int nrows, const int ncol
)
{
    for (int irow = 0; irow < nrows; irow++)
        memcpy(dst + irow * ldd, src + irow * lds, DBL_SIZE * ncol);
} 

double get_wtime_sec()
{
    double sec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}

int block_spos(const int nblk, const int iblk, const int len)
{
    int bs0 = len / nblk;
    int rem = len % nblk;
    int bs1 = bs0 + 1;
    if (iblk < rem) return iblk * bs1;
    else return (iblk * bs0 + rem);
}

// For debug
void print_dbl_mat(double *mat, const int ldm, const int nrow, const int ncol, const char *mat_name)
{
    printf("%s:\n", mat_name);
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < ncol; j++) 
        {
            int idx = i * ldm + j;
            double x = mat[idx];
            if (x >= 0.0) printf(" ");
            printf("%.7lf\t", x);
        }
        printf("\n");
    }
    printf("\n");
}
