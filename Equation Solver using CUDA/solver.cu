/* 
Code for the equation solver. 
Author: Harshvardhan Agrawal
*/

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "grid.h" // This file defines the grid data structure
#include <sys/time.h>
// includes, kernels
#include "solver_kernel.cu"

extern "C" void compute_gold(GRID_STRUCT *);

struct timeval start,stop,start_gpu,stop_gpu,start_gpu_texture,stop_gpu_texture;

float gpu_run_time_global,gpu_run_time_texture;

/* This function prints the grid on the screen */
void 
display_grid(GRID_STRUCT *my_grid)
{
	for(int i = 0; i < my_grid->dimension; i++)
		for(int j = 0; j < my_grid->dimension; j++)
			printf("%f \t", my_grid->element[i * my_grid->dimension + j]);
   		
		printf("\n");
}


/* This function prints out statistics for the converged values, including min, max, and average. */
void 
print_statistics(GRID_STRUCT *my_grid)
{
		// Print statistics for the CPU grid
		float min = INFINITY;
		float max = 0.0;
		double sum = 0.0; 
		for(int i = 0; i < my_grid->dimension; i++){
			for(int j = 0; j < my_grid->dimension; j++){
				sum += my_grid->element[i * my_grid->dimension + j]; // Compute the sum
				if(my_grid->element[i * my_grid->dimension + j] > max) max = my_grid->element[i * my_grid->dimension + j]; // Determine max
				if(my_grid->element[i * my_grid->dimension + j] < min) min = my_grid->element[i * my_grid->dimension + j]; // Determine min
				 
			}
		}

	printf("AVG: %f \n", sum/(float)my_grid->num_elements);
	printf("MIN: %f \n", min);
	printf("MAX: %f \n", max);

	printf("\n");
}


/* Calculate the differences between grid elements for the various implementations. */
void compute_grid_differences(GRID_STRUCT *grid_1, GRID_STRUCT *grid_2)
{
    float diff;
    int dimension = grid_1->dimension;
    int num_elements = dimension*dimension;

    diff = 0.0;
    for(int i = 0; i < grid_1->dimension; i++){
        for(int j = 0; j < grid_1->dimension; j++){
            diff += fabsf(grid_1->element[i * dimension + j] - grid_2->element[i * dimension + j]);
        }
    }
    printf("Average difference in grid elements for Gauss Seidel and Jacobi methods = %f. \n", \
            diff/num_elements);

}



/* This function creates a grid of random floating point values bounded by UPPER_BOUND_ON_GRID_VALUE */
void 
create_grids(GRID_STRUCT *grid_for_cpu, GRID_STRUCT *grid_for_gpu)
{
	printf("Creating a grid of dimension %d x %d. \n", grid_for_cpu->dimension, grid_for_cpu->dimension);
	grid_for_cpu->element = (float *)malloc(sizeof(float) * grid_for_cpu->num_elements);
	grid_for_gpu->element = (float *)malloc(sizeof(float) * grid_for_gpu->num_elements);


	srand((unsigned)time(NULL)); // Seed the the random number generator 
	
	float val;
	for(int i = 0; i < grid_for_cpu->dimension; i++)
		for(int j = 0; j < grid_for_cpu->dimension; j++){
			val =  ((float)rand()/(float)RAND_MAX) * UPPER_BOUND_ON_GRID_VALUE; // Obtain a random value
			grid_for_cpu->element[i * grid_for_cpu->dimension + j] = val; 	
			grid_for_gpu->element[i * grid_for_gpu->dimension + j] = val; 				
		}
}

GRID_STRUCT* 
AllocateDeviceGrid(GRID_STRUCT* my_grid)                        /* Allocate a device grid of same size as my_grid. */
{
	GRID_STRUCT* d_grid;
	d_grid = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT));
	int size = my_grid->num_elements*sizeof(float);
	cudaMalloc((void**)&d_grid->element,size);
	printf("Allocated memory for elements\n");
	return d_grid;
}

void 
CopyToDeviceGrid(GRID_STRUCT* d_grid, GRID_STRUCT* my_grid)      /* Copy a host grid to a device grid. */
{
	printf("Entering copy to device grid \n");
	d_grid->dimension = my_grid->dimension;
	d_grid->num_elements = my_grid->num_elements;
	int size = my_grid->num_elements*sizeof(float);
	cudaMemcpy(d_grid->element, my_grid->element,size, cudaMemcpyHostToDevice);
}

void CopyFromDeviceGrid(GRID_STRUCT* my_grid,GRID_STRUCT* d_grid){
	int size = my_grid->num_elements*sizeof(float);
	cudaMemcpy(my_grid->element, d_grid->element, size, cudaMemcpyDeviceToHost);
}

void FreeDeviceGrid(GRID_STRUCT* d_grid){
	cudaFree(d_grid->element);
	d_grid->element = NULL;
}

void 
checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) 
	{
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}						 
}

/* Edit this function skeleton to solve the equation on the device. Store the results back in the my_grid->element data structure for comparison with the CPU result. */
void 
compute_on_device(GRID_STRUCT *my_grid)
{
	printf("Entering compute on device using global memory\n");
	GRID_STRUCT* d_grid1 = AllocateDeviceGrid(my_grid);                    /* Load my_grid to the device. */
	CopyToDeviceGrid(d_grid1, my_grid);

	GRID_STRUCT* d_grid2 = AllocateDeviceGrid(my_grid);
	CopyToDeviceGrid(d_grid2, my_grid);

	float *h_diff;
	h_diff = (float *)malloc(sizeof(float));
	*h_diff = GRID_DIMENSION*GRID_DIMENSION;
	float *d_diff;
//	int size_temp = GRID_DIMENSION*GRID_DIMENSION*sizeof(float);
//	h_diff = (float *)malloc(size_temp);
	cudaMalloc((void **)&d_diff,sizeof(float));
/*	for (int i=0;i<GRID_DIMENSION*GRID_DIMENSION;i++){
		h_diff[i]=0.0;
	}
	cudaMemcpy(d_diff,h_diff,size_temp,cudaMemcpyHostToDevice);
*/
	// Allocate space for the lock on the GPU and initialize it
	int *mutex_on_device=NULL;
	cudaMalloc((void**)&mutex_on_device,sizeof(int));
	cudaMemset(mutex_on_device,0,sizeof(int));

	// Set up the execution grid
	dim3 threads(TILE_SIZE,TILE_SIZE);
	dim3 grid(GRID_DIMENSION/threads.x,GRID_DIMENSION/threads.y);
	int done=0;
	int numiters=0;
	
	printf("Launching kernel loop\n");
	// Launch the kernel
	gettimeofday(&start_gpu,NULL);
	while(!done){
		cudaMemset(d_diff,0.0f,sizeof(float));
		numiters++;
		solver_kernel_naive<<< grid, threads>>>(d_grid1->element,d_grid2->element,GRID_DIMENSION,d_diff,mutex_on_device);
		cudaThreadSynchronize();
		checkCUDAError("Error in kernel");   
		cudaMemcpy(h_diff,d_diff,sizeof(float),cudaMemcpyDeviceToHost);
	/*	for(int k=0;k<GRID_DIMENSION*GRID_DIMENSION;k++){
			diff = diff + h_diff[k]; 	
		}*/
		if (*h_diff/(float)(GRID_DIMENSION*GRID_DIMENSION) < (float)TOLERANCE) done=1;
		printf("numiters:%d done=%d diff=%f\n",numiters,done,*h_diff);
	}
	gettimeofday(&stop_gpu,NULL);
	
	gpu_run_time_global = (float)(stop_gpu.tv_sec - start_gpu.tv_sec + (stop_gpu.tv_usec - start_gpu.tv_usec)/(float)(1000000));
	
	CopyFromDeviceGrid(my_grid,d_grid1);
	
	cudaFree(mutex_on_device);
	cudaFree(d_diff);
	FreeDeviceGrid(d_grid2);
	FreeDeviceGrid(d_grid1);
	free(h_diff);
	free(d_grid2);
	free(d_grid1);
}

void 
compute_on_device_texture(GRID_STRUCT *my_grid_1)
{
	printf("Entering compute on device using texture memory\n");
	GRID_STRUCT* d_grid1 = AllocateDeviceGrid(my_grid_1);                    /* Load my_grid to the device. */
	CopyToDeviceGrid(d_grid1, my_grid_1);

	GRID_STRUCT* d_grid2 = AllocateDeviceGrid(my_grid_1);
	CopyToDeviceGrid(d_grid2, my_grid_1);

	float *h_diff;
	h_diff = (float *)malloc(sizeof(float));
	*h_diff = GRID_DIMENSION*GRID_DIMENSION;
	
	float *d_diff;
	cudaMalloc((void **)&d_diff,sizeof(float));
	

	// Allocate space for the lock on the GPU and initialize it
	int *mutex_on_device=NULL;
	cudaMalloc((void**)&mutex_on_device,sizeof(int));
	cudaMemset(mutex_on_device,0,sizeof(int));

	//Bind grid1 and grid2 elements to textures
/*	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaChannelFormatDesc desc1 = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, grid1_on_tex_2D, d_grid1->element, desc, d_grid1->dimension, d_grid1->dimension, d_grid1->dimension*sizeof(float));
	cudaBindTexture2D(NULL, grid2_on_tex_2D, d_grid2->element, desc1, d_grid2->dimension, d_grid2->dimension, d_grid2->dimension*sizeof(float));
*/
	cudaBindTexture(NULL, grid1_on_tex, d_grid1->element, d_grid1->num_elements*sizeof(float));
	cudaBindTexture(NULL, grid2_on_tex, d_grid2->element, d_grid2->num_elements*sizeof(float));

	// Set up the execution grid
	dim3 threads(TILE_SIZE,TILE_SIZE);
	dim3 grid(GRID_DIMENSION/threads.x,GRID_DIMENSION/threads.y);
	int done=0;
	int numiters=0;

	printf("Launching kernel loop\n");
	// Launch the kernel
	gettimeofday(&start_gpu_texture,NULL);
	while(!done){
		cudaMemset(d_diff,0.0f,sizeof(float));
		numiters++;
		solver_kernel_optimized<<< grid, threads>>>(d_grid1->element,d_grid2->element,GRID_DIMENSION,d_diff,mutex_on_device);
		cudaThreadSynchronize();
		checkCUDAError("Error in kernel");   
		cudaMemcpy(h_diff,d_diff,sizeof(float),cudaMemcpyDeviceToHost);
	/*	for(int k=0;k<GRID_DIMENSION*GRID_DIMENSION;k++){
			diff = diff + h_diff[k]; 	
		}*/
		if (*h_diff/(float)(GRID_DIMENSION*GRID_DIMENSION) < (float)TOLERANCE) done=1;
		printf("numiters:%d done=%d diff=%f\n",numiters,done,*h_diff);
	}
	gettimeofday(&stop_gpu_texture,NULL);
	
	gpu_run_time_texture = (float)(stop_gpu_texture.tv_sec - start_gpu_texture.tv_sec + (stop_gpu_texture.tv_usec - start_gpu_texture.tv_usec)/(float)(1000000));
	
	CopyFromDeviceGrid(my_grid_1,d_grid1);
	
	
	cudaFree(d_diff);
	cudaUnbindTexture(grid1_on_tex);
	cudaUnbindTexture(grid2_on_tex);
	FreeDeviceGrid(d_grid2);
	FreeDeviceGrid(d_grid1);
	free(h_diff);
	free(d_grid2);
	free(d_grid1);
}

/* The main function */
int 
main(int argc, char **argv)
{	
	/* Generate the grid */
	GRID_STRUCT *grid_for_cpu = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); // The grid data structure
	GRID_STRUCT *grid_for_gpu = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); // The grid data structure
	GRID_STRUCT *grid_for_gpu_texture = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT));

	grid_for_cpu->dimension = GRID_DIMENSION;
	grid_for_cpu->num_elements = grid_for_cpu->dimension * grid_for_cpu->dimension;
	grid_for_gpu->dimension = GRID_DIMENSION;
	grid_for_gpu->num_elements = grid_for_gpu->dimension * grid_for_gpu->dimension;
	grid_for_gpu_texture->dimension = GRID_DIMENSION;
	grid_for_gpu_texture->num_elements = grid_for_gpu_texture->dimension * grid_for_gpu_texture->dimension;

	grid_for_gpu_texture->element = (float *)malloc(sizeof(float)*grid_for_gpu_texture->num_elements);
 	create_grids(grid_for_cpu, grid_for_gpu); // Create the grids and populate them with the same set of random values
	printf("Creating grid for texture memory\n");	
	
	for (int i=0;i<GRID_DIMENSION;i++) {
		for(int j=0;j<GRID_DIMENSION;j++){
			grid_for_gpu_texture->element[i*GRID_DIMENSION + j] = grid_for_gpu->element[i*GRID_DIMENSION + j];
		}
	}
	printf("Using the cpu to solve the grid. \n");
	gettimeofday(&start,NULL);
	compute_gold(grid_for_cpu);  // Use CPU to solve 
	gettimeofday(&stop,NULL);
	float cpu_run_time = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)(1000000));
	
	// Use the GPU to solve the equation
	compute_on_device(grid_for_gpu);
	
	compute_on_device_texture(grid_for_gpu_texture);	
	
	// Print key statistics for the converged values
	printf("CPU: \n");
	print_statistics(grid_for_cpu);

	printf("GPU: \n");
	print_statistics(grid_for_gpu);

	printf("GPU with texture: \n");
	print_statistics(grid_for_gpu_texture);	
    /* Compute grid differences. */
    	compute_grid_differences(grid_for_cpu, grid_for_gpu);
	compute_grid_differences(grid_for_cpu,grid_for_gpu_texture);
	printf("CPU run time: %0.4fs\n",cpu_run_time);
	printf("GPU run time with global memory: %0.4fs\n",gpu_run_time_global);
	printf("Speedup (global memory): %0.4f\n",(cpu_run_time/gpu_run_time_global));

	printf("GPU run time with texture memory: %0.4fs\n",gpu_run_time_texture);
	printf("Speedup (texture memory): %0.4f\n",(cpu_run_time/gpu_run_time_texture));	
	
	free((void *)grid_for_cpu->element);	
	free((void *)grid_for_cpu); // Free the grid data structure 
	
	free((void *)grid_for_gpu->element);	
	free((void *)grid_for_gpu); // Free the grid data structure 

	free((void *)grid_for_gpu_texture->element);
	free((void *)grid_for_gpu_texture);
	exit(0);
}
