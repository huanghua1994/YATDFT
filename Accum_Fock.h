#ifndef _YATSCF_ACCUM_FOCK_H_
#define _YATSCF_ACCUM_FOCK_H_

#include "TinySCF.h"

#define ACCUM_FOCK_IN_PARAM TinySCF_t TinySCF, int tid, int M, int N, int P, int Q, \
                            double *ERI, int load_P, int write_P, \
                            double *FM_strip_blocks, int FM_strip_offset, \
                            double *FN_strip_blocks, int FN_strip_offset

void Accum_Fock(ACCUM_FOCK_IN_PARAM);

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