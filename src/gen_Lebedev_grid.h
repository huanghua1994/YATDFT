#ifndef __GEN_LEBEDEV_GRID_H__
#define __GEN_LEBEDEV_GRID_H__

#ifdef __cplusplus
extern "C" {
#endif

// Generate Lebedev grids for sphere integral
// Input parameter:
//   nPoints : Number pf Lebedev grid points
// Output parameters:
//   Out      : Size nPoint-by-4, each row stores x/y/z coordinate and weight
//   <return> : Number of generated grids, should equal to nPoints, 0 means 
//              input nPoint value is not supported
int gen_Lebedev_grid(const int nPoints, double *Out);

#ifdef __cplusplus
}
#endif

#endif

