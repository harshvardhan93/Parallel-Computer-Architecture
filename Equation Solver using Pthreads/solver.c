/* Code for the equation solver. 
 * Author: Harshvardhan Agrawal
 *
 * Compile as follows:
 * gcc -o solver solver.c solver_gold.c -std=c99 -lm -lpthread 
 */

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include "grid.h" 
#include <sys/times.h>
#define NUM_THREADS 16
#define DIMENSION 8192
GRID_STRUCT *grid_1;
GRID_STRUCT *grid_2;
GRID_STRUCT *grid_3;
GRID_STRUCT *grid_4;
GRID_STRUCT *grid_5;

extern int compute_gold(GRID_STRUCT *);
int compute_using_pthreads_jacobi(GRID_STRUCT *);
int compute_using_pthreads_red_black(GRID_STRUCT *);
void compute_grid_differences(GRID_STRUCT *, GRID_STRUCT *, GRID_STRUCT *);
int red_black_compute(void *);
void * jacobi(void *);
void * redblack(void *);
int chunksize;
double final_diff;
pthread_mutex_t difference;
pthread_barrier_t barrier;
pthread_barrier_t barrier1;
int iterator_jacobi;
int flag_jacobi;
int iterator_redblack;
int flag_redblack;

/* This function prints the grid on the screen. */
void 
display_grid(GRID_STRUCT *my_grid)
{
    for(int i = 0; i < my_grid->dimension; i++)
        for(int j = 0; j < my_grid->dimension; j++)
            printf("%f \t", my_grid->element[i * my_grid->dimension + j]);
   		
    printf("\n");
}


/* Print out statistics for the converged values, including min, max, and average. */
void 
print_statistics(GRID_STRUCT *my_grid)
{
    float min = INFINITY;
    float max = 0.0;
    double sum = 0.0; 
    
    for(int i = 0; i < my_grid->dimension; i++){
        for(int j = 0; j < my_grid->dimension; j++){
            sum += my_grid->element[i * my_grid->dimension + j];
           
            if(my_grid->element[i * my_grid->dimension + j] > max) 
                max = my_grid->element[i * my_grid->dimension + j];

				if(my_grid->element[i * my_grid->dimension + j] < min) 
                    min = my_grid->element[i * my_grid->dimension + j];
				 
        }
    }

    printf("AVG: %f \n", sum/(float)my_grid->num_elements);
	printf("MIN: %f \n", min);
	printf("MAX: %f \n", max);

	printf("\n");
}

/* Calculate the differences between grid elements for the various implementations. */
void compute_grid_differences(GRID_STRUCT *grid_1, GRID_STRUCT *grid_2, GRID_STRUCT *grid_3)
{
    float diff_12, diff_13;
    int dimension = grid_1->dimension;
    int num_elements = dimension*dimension;

    diff_12 = 0.0;
    diff_13 = 0.0;
    for(int i = 0; i < grid_1->dimension; i++){
        for(int j = 0; j < grid_1->dimension; j++){
            diff_12 += fabsf(grid_1->element[i * dimension + j] - grid_3->element[i * dimension + j]);
            diff_13 += fabsf(grid_1->element[i * dimension + j] - grid_2->element[i * dimension + j]);
        }
    }
    printf("Average difference in grid elements for Gauss Seidel and Red-Black methods = %f. \n", \
            diff_12/num_elements);

    printf("Average difference in grid elements for Gauss Seidel and Jacobi methods = %f. \n", \
            diff_13/num_elements);


}

/* Create a grid of random floating point values bounded by UPPER_BOUND_ON_GRID_VALUE */
void 
create_grids(GRID_STRUCT *grid_1, GRID_STRUCT *grid_2, GRID_STRUCT *grid_3,GRID_STRUCT *grid_4,GRID_STRUCT *grid5)
{
	printf("Creating a grid of dimension %d x %d. \n", grid_1->dimension, grid_1->dimension);
	grid_1->element = (float *)malloc(sizeof(float) * grid_1->num_elements);
	grid_2->element = (float *)malloc(sizeof(float) * grid_2->num_elements);
	grid_3->element = (float *)malloc(sizeof(float) * grid_3->num_elements);
	grid_4->element = (float *)malloc(sizeof(float) * grid_4->num_elements);
	grid_5->element = (float *)malloc(sizeof(float) * grid_5->num_elements);
	srand((unsigned)time(NULL));
	
	float val;
	for(int i = 0; i < grid_1->dimension; i++)
		for(int j = 0; j < grid_1->dimension; j++){
			val =  ((float)rand()/(float)RAND_MAX) * UPPER_BOUND_ON_GRID_VALUE;
			grid_1->element[i * grid_1->dimension + j] = val; 	
			grid_2->element[i * grid_2->dimension + j] = val; 
			grid_3->element[i * grid_3->dimension + j] = val; 
			grid_4->element[i * grid_4->dimension + j] = val;
			grid_5->element[i * grid_5->dimension + j] = val;	
	}
}

/* Edit this function to use the jacobi method of solving the equation. The final result should 
 * be placed in the grid_2 data structure */
int 
compute_using_pthreads_jacobi(GRID_STRUCT *grid_2)
{
//	printf("\nComputing Jacobi using pthreads now!");		
	pthread_t thread_id[NUM_THREADS]; // Data structure to store the thread IDs
	pthread_mutex_init(&difference,NULL);
	pthread_barrier_init(&barrier,NULL,NUM_THREADS);
	for(int i = 0; i < NUM_THREADS; i++) {
		if((pthread_create(&thread_id[i],NULL, jacobi, (void *)i))!=0){
			printf("\n Error creating threads!");		
			exit(0);
		}
	}
	for(int i=0;i<NUM_THREADS;i++)
	{
	pthread_join(thread_id[i],NULL);
	}		
//	printf("\n Exiting compute pthreads!");
	return iterator_jacobi;
}

void * jacobi(void * id)
{	
	printf("\nThread %d is starting jacobi",(int)id);
	int i,j;
  	int done=0;//if golabl diff/num of elements is less than tolerance, done = 1
  	float diff;// local thread difference sum
 	int thread_id=(int)id;
  	int offset=thread_id*chunksize;
//	printf("\n Thread % d Chunksize is %d",thread_id,chunksize);
//	printf("\n Thread %d Offset is %d ",thread_id,offset);
//	printf("\n Thread %d is entering while loop",thread_id);

  	while(!done){ 

	final_diff=0;
      	diff=0;
      	flag_jacobi=1;

	if(thread_id==0) {
	for(i=1; i<chunksize;i++){
      
		for(j=1;j<DIMENSION-1;j++) {
	 
			 grid_2->element[i*DIMENSION+j]=0.20*(grid_4->element[i * grid_4->dimension + j] + \
                        	               grid_4->element[(i - 1) * grid_4->dimension + j] +\
                                	       grid_4->element[(i + 1) * grid_4->dimension + j] +\
                                      	 	grid_4->element[i * grid_4->dimension + (j + 1)] +\
                                       		grid_4->element[i * grid_4->dimension + (j - 1)]);
	  		diff += fabs(grid_2->element[i * DIMENSION + j] - grid_4->element[i * grid_4->dimension + j]);
		}
    	}
    	} else if(thread_id==NUM_THREADS-1) {
      		for(i=offset;i<DIMENSION-1;i++) {
	  		for(j=1;j<DIMENSION-1;j++) {
	      			grid_2->element[i*DIMENSION+j]=0.20*(grid_4->element[i * grid_4->dimension + j] + \
                                       	grid_4->element[(i - 1) * grid_4->dimension + j] +\
                                       	grid_4->element[(i + 1) * grid_4->dimension + j] +\
                                       	grid_4->element[i * grid_4->dimension + (j + 1)] +\
                                       	grid_4->element[i * grid_4->dimension + (j - 1)]);
	  			diff += fabs(grid_2->element[i * DIMENSION + j] - grid_4->element[i * grid_4->dimension + j]);
	    		}		
		}		
    	}
  	else {
    		for(i=offset;i< offset+ chunksize;i++) {
	  		for(j=1;j<DIMENSION-1;j++){
	      			grid_2->element[i*DIMENSION+j]=0.20*(grid_4->element[i * grid_4->dimension + j] + \
                                       grid_4->element[(i - 1) * grid_4->dimension + j] +\
                                       grid_4->element[(i + 1) * grid_4->dimension + j] +\
                                       grid_4->element[i * grid_4->dimension + (j + 1)] +\
                                       grid_4->element[i * grid_4->dimension + (j - 1)]);
	  			diff += fabs(grid_2->element[i * DIMENSION + j] - grid_4->element[i * grid_4->dimension + j]);
	    		}
		}
	}
//	printf("\n Thread %d Approaching first barrier!",thread_id);
	pthread_barrier_wait(&barrier);
	pthread_mutex_lock(&difference);
	final_diff+=diff;
	if(flag_jacobi==1) {
	printf("\n Thread %d is incrementing iterator to %d",thread_id,iterator_jacobi);

		iterator_jacobi++;
     		flag_jacobi=0;
   	}
	pthread_mutex_unlock(&difference);
/*	if(thread_id==0)
	{
		for(i=1;i<DIMENSION-1;i++)
	{	
		for(j=1;j<DIMENSION-1;j++)
	{
		final_diff+=fabs(grid_2->element[i * DIMENSION +j]-grid_4->element[i* DIMENSION +j]);		
	}
	}
	}
*/
//printf("\n Thread %d is approaching second barrier!", thread_id);	
	pthread_barrier_wait(&barrier);
	if(thread_id==0){
	  //		printf("\n Thread 0 is updating grid 4");
 		for(i=0;i<DIMENSION;i++) {
     			for(j=0;j<DIMENSION;j++){
	 			grid_4->element[i* DIMENSION + j]=grid_2->element[i* DIMENSION + j];
       			}
   		}
   	}
 //	printf("\n Thread %d is aprroaching third barrier!",thread_id);
	pthread_barrier_wait(&barrier);
	//printf("\n Final diff is %lf",final_diff);
	double temp = (double)final_diff;
	double temp1 = (double)(DIMENSION*DIMENSION);	
	double temp2 = (temp/temp1);
//	printf("\n Temp2 is %lf",temp2);
//	printf("\n Tolerance %f",(float)TOLERANCE);
	if(temp2 < (double)TOLERANCE) {
            	//printf("\n I am about to get done!");
		done = 1;
	}
	//	if (done==1)
	  //	printf("\n I'm done!");
    	pthread_barrier_wait(&barrier);
	}
 pthread_exit(NULL);
}

/* Edit this function to use the red-black method of solving the equation. The final result 
 * should be placed in the grid_3 data structure */
int 
compute_using_pthreads_red_black(GRID_STRUCT *grid_3)
{
    //    printf("\nComputing Red Black using pthreads now!");
        pthread_t thread_id[NUM_THREADS]; // Data structure to store the thread IDs
        pthread_mutex_init(&difference,NULL);
        pthread_barrier_init(&barrier1,NULL,NUM_THREADS);
        for(int i = 0; i < NUM_THREADS; i++) {
                if((pthread_create(&thread_id[i],NULL, redblack, (void *)i))!=0){
                        printf("\n Error creating threads!");
                        exit(0);
                }
        }
	for(int i=0;i<NUM_THREADS;i++){
		pthread_join(thread_id[i],NULL);
	}
//        printf("\n Exiting compute pthreads!");
        return iterator_redblack;

}

void * redblack (void * id){
//	printf("\nThread %d is starting Redblack",(int)id);
	int i,j;
  	int done=0;//if golabl diff/num of elements is less than tolerance, done = 1
  	float diff;// local thread difference sum
 	int thread_id=(int)id;
  	int offset=thread_id*chunksize;
//	printf("\n Thread % d Chunksize is %d",thread_id,chunksize);
//	printf("\n Thread %d Offset is %d ",thread_id,offset);
//	printf("\n Thread %d is entering while loop",thread_id);

  	while(!done){ 

	final_diff=0;
      	diff=0;
      	flag_redblack=1;

	if(thread_id==0) {
	for(i=1; i<chunksize;i++){
		for(j=1;j<DIMENSION-1;j++) {
	 		if((i+j)%2==0) {
			 grid_3->element[i*DIMENSION+j]=0.20*(grid_5->element[i * grid_5->dimension + j] + \
                        	               grid_5->element[(i - 1) * grid_5->dimension + j] +\
                                	       grid_5->element[(i + 1) * grid_5->dimension + j] +\
                                      	 	grid_5->element[i * grid_5->dimension + (j + 1)] +\
                                       		grid_5->element[i * grid_5->dimension + (j - 1)]);
	  		diff += fabs(grid_3->element[i * DIMENSION + j] - grid_5->element[i * grid_5->dimension + j]);
			}
		}
    	}
    	} else if(thread_id==NUM_THREADS-1) {
      		for(i=offset;i<DIMENSION-1;i++) {
	  		for(j=1;j<DIMENSION-1;j++) {
				if((i+j)%2==0) {
	      			grid_3->element[i*DIMENSION+j]=0.20*(grid_5->element[i * grid_5->dimension + j] + \
                                       	grid_5->element[(i - 1) * grid_5->dimension + j] +\
                                       	grid_5->element[(i + 1) * grid_5->dimension + j] +\
                                       	grid_5->element[i * grid_5->dimension + (j + 1)] +\
                                       	grid_5->element[i * grid_5->dimension + (j - 1)]);
	  			diff += fabs(grid_3->element[i * DIMENSION + j] - grid_5->element[i * grid_5->dimension + j]);
	    			}
			}		
		}		
    	}
  	else {
    		for(i=offset;i< offset+ chunksize;i++) {
	  		for(j=1;j<DIMENSION-1;j++){
				if((i+j)%2==0){
	      			grid_3->element[i*DIMENSION+j]=0.20*(grid_5->element[i * grid_5->dimension + j] + \
                                       grid_5->element[(i - 1) * grid_5->dimension + j] +\
                                       grid_5->element[(i + 1) * grid_5->dimension + j] +\
                                       grid_5->element[i * grid_5->dimension + (j + 1)] +\
                                       grid_5->element[i * grid_5->dimension + (j - 1)]);
	  			diff += fabs(grid_3->element[i * DIMENSION + j] - grid_5->element[i * grid_5->dimension + j]);
				}
	    		}

		}
	}
	pthread_barrier_wait(&barrier1);	
	if(thread_id==0){
//		printf("\n Thread 0 is updating grid 5");
 		for(i=0;i<DIMENSION;i++) {
     			for(j=0;j<DIMENSION;j++){
	 			grid_5->element[i* DIMENSION + j]=grid_3->element[i* DIMENSION + j];
       			}
   		}
   	}
	pthread_barrier_wait(&barrier1);
	if(thread_id==0) {
	for(i=1; i<chunksize;i++){
		for(j=1;j<DIMENSION-1;j++) {
	 		if((i+j)%2==1) {
			 grid_3->element[i*DIMENSION+j]=0.20*(grid_5->element[i * grid_5->dimension + j] + \
                        	               grid_5->element[(i - 1) * grid_5->dimension + j] +\
                                	       grid_5->element[(i + 1) * grid_5->dimension + j] +\
                                      	 	grid_5->element[i * grid_5->dimension + (j + 1)] +\
                                       		grid_5->element[i * grid_5->dimension + (j - 1)]);
	  		diff += fabs(grid_3->element[i * DIMENSION + j] - grid_5->element[i * grid_5->dimension + j]);
			}
		}
    	}
    	} else if(thread_id==NUM_THREADS-1) {
      		for(i=offset;i<DIMENSION-1;i++) {
	  		for(j=1;j<DIMENSION-1;j++) {
				if((i+j)%2==1) {
	      			grid_3->element[i*DIMENSION+j]=0.20*(grid_5->element[i * grid_5->dimension + j] + \
                                       	grid_5->element[(i - 1) * grid_5->dimension + j] +\
                                       	grid_5->element[(i + 1) * grid_5->dimension + j] +\
                                       	grid_5->element[i * grid_5->dimension + (j + 1)] +\
                                       	grid_5->element[i * grid_5->dimension + (j - 1)]);
	  			diff += fabs(grid_3->element[i * DIMENSION + j] - grid_5->element[i * grid_5->dimension + j]);
	    			}
			}		
		}		
    	}
  	else {
    		for(i=offset;i< offset+ chunksize;i++) {
	  		for(j=1;j<DIMENSION-1;j++){
				if((i+j)%2==1){
	      			grid_3->element[i*DIMENSION+j]=0.20*(grid_5->element[i * grid_5->dimension + j] + \
                                       grid_5->element[(i - 1) * grid_5->dimension + j] +\
                                       grid_5->element[(i + 1) * grid_5->dimension + j] +\
                                       grid_5->element[i * grid_5->dimension + (j + 1)] +\
                                       grid_5->element[i * grid_5->dimension + (j - 1)]);
	  			diff += fabs(grid_3->element[i * DIMENSION + j] - grid_5->element[i * grid_5->dimension + j]);
				}
	    		}

		}
	}

//	printf("\n Thread %d Approaching first barrier!",thread_id);
	pthread_barrier_wait(&barrier1);
	pthread_mutex_lock(&difference);
	final_diff+=diff;
	if(flag_redblack==1) {
	printf("\n Thread %d is incrementing iterator to %d",thread_id,iterator_redblack);

		iterator_redblack++;
     		flag_redblack=0;
   	}
	pthread_mutex_unlock(&difference);
/*	if(thread_id==0)
	{
		for(i=1;i<DIMENSION-1;i++)
	{	
		for(j=1;j<DIMENSION-1;j++)
	{
		final_diff+=fabs(grid_2->element[i * DIMENSION +j]-grid_4->element[i* DIMENSION +j]);		
	}
	}
	}
*/
//	printf("\n Thread %d is approaching second barrier!", thread_id);	
	pthread_barrier_wait(&barrier1);
	if(thread_id==0){
//		printf("\n Thread 0 is updating grid 5");
 		for(i=0;i<DIMENSION;i++) {
     			for(j=0;j<DIMENSION;j++){
	 			grid_5->element[i* DIMENSION + j]=grid_3->element[i* DIMENSION + j];
       			}
   		}
   	}
 //	printf("\n Thread %d is aprroaching third barrier!",thread_id);
	pthread_barrier_wait(&barrier1);
//	printf("\n Final diff is %lf",final_diff);
	double temp = (double)final_diff;
	double temp1 = (double)(DIMENSION*DIMENSION);	
	double temp2 = (temp/temp1);
//	printf("\n Temp2 is %lf",temp2);
//	printf("\n Tolerance %f",(float)TOLERANCE);
	if(temp2 < (double)TOLERANCE) {
            	//printf("\n I am about to get done!");
		done = 1;
	}
	//if (done==1)
	//	printf("\n I'm done!");
    	pthread_barrier_wait(&barrier1);
	}
 pthread_exit(NULL);


}
		
/* The main function */
int 
main(int argc, char **argv)
{	
	/* Generate the grids and populate them with the same set of random values. */
	grid_1 = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); 
	grid_2 = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); 
	grid_3 = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); 
	grid_4 = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT));
	grid_5 = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT));	

	grid_1->dimension = GRID_DIMENSION;
	grid_1->num_elements = grid_1->dimension * grid_1->dimension;
	grid_2->dimension = GRID_DIMENSION;
	grid_2->num_elements = grid_2->dimension * grid_2->dimension;
	grid_3->dimension = GRID_DIMENSION;
	grid_3->num_elements = grid_3->dimension * grid_3->dimension;
	grid_4->dimension = GRID_DIMENSION;
	grid_4->num_elements = grid_4->dimension * grid_4->dimension;
	grid_5->dimension = GRID_DIMENSION;
	grid_5->num_elements = grid_5->dimension * grid_5->dimension;
	struct timeval start,start1,start2,stop,stop1,stop2;
 	create_grids(grid_1, grid_2, grid_3,grid_4,grid_5);
	chunksize=(int)floor((float)grid_1->dimension/(float)NUM_THREADS);
	final_diff=0;
	/* Compute the reference solution using the single-threaded version. */
	printf("Using the single threaded version to solve the grid. \n");
	gettimeofday(&start,NULL);
	int num_iter = compute_gold(grid_1);
	gettimeofday(&stop,NULL);
	float serialtime=(float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("Convergence achieved after %d iterations. \n", num_iter);

	/* Use pthreads to solve the equation uisng the red-black parallelization technique. */
	printf("Using pthreads to solve the grid using the red-black parallelization method. \n");
	gettimeofday(&start1,NULL);
	num_iter = compute_using_pthreads_red_black(grid_3);
	gettimeofday(&stop1,NULL);
	float redtime=(float)(stop1.tv_sec - start1.tv_sec + (stop1.tv_usec - start1.tv_usec)/(float)1000000);
	printf("Convergence achieved after %d iterations. \n", num_iter);
	
	/* Use pthreads to solve the equation using the jacobi method in parallel. */
	printf("Using pthreads to solve the grid using the jacobi method. \n");
	gettimeofday(&start2,NULL);
	num_iter = compute_using_pthreads_jacobi(grid_2);
	gettimeofday(&stop2,NULL);
	float jacobitime=(float)(stop2.tv_sec - start2.tv_sec + (stop2.tv_usec - start2.tv_usec)/(float)1000000);
printf("\nConvergence achieved after %d iterations. \n", num_iter);
	/* Print key statistics for the converged values. */
	printf("\n");
	printf("Reference: \n");
	print_statistics(grid_1);

	printf("Red-black: \n");
	print_statistics(grid_3);
		
	printf("Jacobi: \n");
	print_statistics(grid_2);
	printf("\n Iterations for Jacobi: %d",iterator_jacobi);
	printf("\n Iterations for Red Black: %d",iterator_redblack);
    /* Compute grid differences. */
	printf("\nCPU run time serial = %0.2f s. \n",(float)( serialtime/1000));
	printf("CPU run time red black = %0.2f s. \n", (float)(redtime/1000));
	printf("CPU run time jacobi = %0.2f s. \n", (float)(jacobitime/1000));	

    compute_grid_differences(grid_1, grid_2, grid_3);


/* Free up the grid data structures. */
	free((void *)grid_1->element);	
	free((void *)grid_1); 
	
	free((void *)grid_2->element);	
	free((void *)grid_2);

	free((void *)grid_3->element);	
	free((void *)grid_3);
	exit(0);
}

