#ifndef __GEN_INT_GRID_H__
#define __GEN_INT_GRID_H__

#ifdef __cplusplus
extern "C" {
#endif

// Generate numerical integral points for XC calculations
// Input parameters:
//   natom    : Number of atoms
//   atom_xyz : Size 3-by-natom, each column is the coordinate of an atom
//   atom_idx : Size natom, atom index (H:1, He:2, Li:3, ...)
// Output parameters:
//   *npoint_   : Total number of integral grid points
//   *int_grid_ : Pointer to allocated space that stores integral grid points.
//                Size 4-by-*npoint_, each column stores x/y/z coordinate and weight
void gen_int_grid(
    const int natom, const double *atom_xyz, const int *atom_idx,
    int *npoint_, double **int_grid_
);

#ifdef __cplusplus
}
#endif

#endif
