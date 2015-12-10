/*  Purpose: Calculate definite integral using trapezoidal rule.
 *
 * Input:   a, b, n
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids.
 *
 * Compile: gcc -o trap trap.c -lpthread -lm
 * Usage:   ./trap
 *
 * Note:    The function f(x) is hardwired.
 * Author: Harshvardhan Agrawal
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <pthread.h>

#define LEFT_ENDPOINT 5
#define RIGHT_ENDPOINT 1000
#define NUM_TRAPEZOIDS 100000000
#define NUM_THREADS 16

void compute_using_pthreads(float, float, int, float);
double compute_gold(float, float, int, float);
void * calc_area(void *);
typedef struct myfun_args{
	int thread_id;
	double start_idx;
	double end_idx;
	float height;
} myfun_args;

myfun_args * my_args;
double * local_areas;
pthread_t my_thread[NUM_THREADS];
pthread_barrier_t barr;
double final_area;

int main(void) 
{
	int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	float h = (b-a)/(float)n; // Height of each trapezoid  
	printf("The height of the trapezoid is %f \n", h);
	struct timeval start,stop;
	gettimeofday(&start,NULL);
	
	double reference = compute_gold(a, b, n, h);

	gettimeofday(&stop,NULL);
   	printf("Reference solution computed on the CPU = %0.2f \n", reference);
	printf("CPU run time = %0.4f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	/* Write this function to complete the trapezoidal on the GPU. */
	
	gettimeofday(&start,NULL);

	compute_using_pthreads(a, b, n, h);
	
	gettimeofday(&stop,NULL);
	printf("\nSolution computed using pthreads = %0.2f \n", final_area);
	printf("CPU run time = %0.4f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
 
}

/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 * Output: (x+1)/sqrt(x*x + x + 1)

 */
float f(float x) {
		  return (x + 1)/sqrt(x*x + x + 1);
}  /* f */

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral 
 */
double compute_gold(float a, float b, int n, float h) {
   double integral;
   int k;

   integral = (f(a) + f(b))/2.0;
   for (k = 1; k <= n-1; k++) {
     integral += f(a+k*h);
   }
   integral = integral*h;

   return integral;
}  

/* Complete this function to perform the trapezoidal rule on the GPU. */
void compute_using_pthreads(float a, float b, int n, float h){
	float temp = fmodf((float)n,(float)NUM_THREADS);
	float num_traps_per_thread = ((float)n - temp)/(float)NUM_THREADS;
	printf("\n Number of threads: %d \n Number of traps: %d\n Num of traps per thread %f",NUM_THREADS,n,num_traps_per_thread);
	local_areas = (double *)malloc(sizeof(double)*NUM_THREADS);
	int i;
	for(i=0;i<NUM_THREADS;i++){
		local_areas[i]=0.0;
	}
	pthread_barrier_init(&barr,NULL,NUM_THREADS);
	for(i=0;i<NUM_THREADS;i++){
		my_args = (myfun_args*)malloc(sizeof(myfun_args));
		my_args->thread_id = i;
		if(i==NUM_THREADS-1){
			my_args->start_idx = a+(num_traps_per_thread*i)*h;
			my_args->end_idx = b;
		}
		else {
			my_args->start_idx = a+(num_traps_per_thread*i)*h;
			my_args->end_idx = a+(num_traps_per_thread*(i+1))*h;
		}
		my_args->height = h;
	
		if(pthread_create(&my_thread[i],NULL,calc_area,(void *)my_args)!=0){
			printf("\nCannot create thread!");
			exit(0);
		}
	}
	pthread_join(my_thread[0],NULL);
	for(i=0;i<NUM_THREADS;i++)
		final_area += local_areas[i];
}


void * calc_area(void *my_args){
	myfun_args *my_args_final = (myfun_args *) my_args;
	int temp_id = my_args_final->thread_id;
	double i = my_args_final->start_idx;
	double loop_end = my_args_final->end_idx - my_args_final->height;
	double sum=0;
	pthread_barrier_wait(&barr);
	double a,b,integral;
	while(i<=loop_end){
		a = i;
		b = i + my_args_final->height;
		integral = (f(a)+f(b))/2.0;
		sum = sum + integral;
		i+=my_args_final->height;
	}
	local_areas[temp_id] = sum*my_args_final->height;
	pthread_barrier_wait(&barr);
}
