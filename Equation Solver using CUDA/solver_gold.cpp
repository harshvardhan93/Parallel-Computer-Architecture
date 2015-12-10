#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "grid.h"

extern "C" void compute_gold(GRID_STRUCT *);


/* This function solves the Gauss-Seidel method on the CPU */
void compute_gold(GRID_STRUCT *my_grid)
{
	int num_iter = 0;
	int done = 0;
	float diff;
	float temp; 
	
	while(!done){ // While we have not converged yet 
		printf("Iteration %d. \n", num_iter);
		diff = 0;	
		for(int i = 1; i < (my_grid->dimension-1); i++){
			for(int j = 1; j < (my_grid->dimension-1); j++){
				temp = my_grid->element[i * my_grid->dimension + j]; 
				my_grid->element[i * my_grid->dimension + j] =\
                                  0.20*(my_grid->element[i * my_grid->dimension + j] +\
                                  my_grid->element[(i - 1) * my_grid->dimension + j] +\
                                  my_grid->element[(i + 1) * my_grid->dimension + j] +\
                                  my_grid->element[i * my_grid->dimension + (j + 1)] +\
                                  my_grid->element[i * my_grid->dimension + (j - 1)]);
				diff = diff + fabsf(my_grid->element[i * my_grid->dimension + j] - temp);  
			}
		}
		/* End of an iteration. Check for convergence */
		num_iter++;

		if((float)diff/((float)(my_grid->dimension*my_grid->dimension)) < (float)TOLERANCE) done = 1;
	}
	printf("Convergence achieved on the CPU after %d iterations. \n", num_iter);
}

