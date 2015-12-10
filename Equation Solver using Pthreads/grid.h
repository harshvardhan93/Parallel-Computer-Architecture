#ifndef __GRID_STRUCT__
#define __GRID_STRUCT__

#define GRID_DIMENSION 8192
#define TOLERANCE 0.01                  /* Tolerance value for convergence. */
#define UPPER_BOUND_ON_GRID_VALUE 100   /* The largest value in the grid. */

typedef struct grid_struct{
	int num_elements;
	int dimension;                      /* Dimension of the grid. */
	float *element;
} GRID_STRUCT;

#endif
