#ifndef _YATSCF_ACC_JKMAT_H_
#define _YATSCF_ACC_JKMAT_H_

#include "TinySCF.h"

#define ACC_JKMAT_IN_PARAM  TinySCF_t TinySCF, int tid, int M, int N, int P, int Q, \
                            double *ERI, int load_P, int write_P, \
                            double *FM_strip_buf, int FM_strip_offset, \
                            double *FN_strip_buf, int FN_strip_offset

// Accumulate an shell quartet ERI tensor to local J and K matrix buffers
void acc_JKmat(ACC_JKMAT_IN_PARAM);

// Accumulate a list of shell quartet ERI tensors to local J and K matrix buffers
void acc_JKmat_with_ket_sp_list(
    TinySCF_t TinySCF, int tid, int M, int N, 
    int *P_list, int *Q_list, int npairs, double *ERIs, int nints,
    double *FM_strip_buf, double *FN_strip_buf,
    int *Mpair_flag, int *Npair_flag
);

static inline void atomic_add_f64(volatile double *global_value, double addend)
{
    uint64_t expected_value, new_value;
    do {
        double old_value = *global_value;
        expected_value = _castf64_u64(old_value);
        new_value = _castf64_u64(old_value + addend);
    } while (!__sync_bool_compare_and_swap((volatile uint64_t*)global_value, expected_value, new_value));
}

static inline void atomic_add_vector(double *dst, double *src, int length)
{
    for (int i = 0; i < length; i++) atomic_add_f64(&dst[i], src[i]);
}

static inline void direct_add_vector(double *dst, double *src, int length)
{
    #pragma omp simd
    for (int i = 0; i < length; i++) dst[i] += src[i];
}

#endif