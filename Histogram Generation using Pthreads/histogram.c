/* Pthread Lab: Histrogram generation
 * Author: Harshvardhan Agrawal
 *
 * compile as follows: 
 * gcc -o histogram histogram.c -std=c99 -lpthread -lm
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <pthread.h>

void run_test(int);
void compute_gold(int *, int *, int, int);
void compute_using_pthreads(int *, int *, int, int);
void check_histogram(int *, int, int);
void * calc_hist (void * );
#define HISTOGRAM_SIZE 500      /* Number of histrogram bins. */
#define NUM_THREADS 16           /* Number of threads. */

typedef struct myfun_args {
	int thread_id;
	int start_idx;
	int end_idx;
} myfun_args;

myfun_args * my_args;
int ** local_hist;
int * input_data;	
pthread_t my_thread[NUM_THREADS];
pthread_barrier_t barr;
int 
main( int argc, char** argv) 
{
	if(argc != 2){
		printf("Usage: histogram <num elements> \n");
		exit(0);	
	}
	int num_elements = atoi(argv[1]);
	run_test(num_elements);
	return 0;
}

void 
run_test(int num_elements) 
{
	float diff;
	int i; 

    /* Allocate memory for the histrogram structures. */
	int *reference_histogram = (int *)malloc(sizeof(int) * HISTOGRAM_SIZE);
	int *histogram_using_pthreads = (int *)malloc(sizeof(int) * HISTOGRAM_SIZE); 
//	int *histogram_using_pthreads;
	/* Generate input data---integer values between 0 and (HISTOGRAM_SIZE - 1). */
    int size = sizeof(int) * num_elements;
	input_data = (int *)malloc(size);

	for(i = 0; i < num_elements; i++)
		input_data[i] = floorf((HISTOGRAM_SIZE - 1) * (rand()/(float)RAND_MAX));

    /* Compute the reference solution on the CPU. */
	printf("Creating the reference histogram. \n"); 
	struct timeval start, stop;	
	gettimeofday(&start, NULL);

	compute_gold(input_data, reference_histogram, num_elements, HISTOGRAM_SIZE);

	gettimeofday(&stop, NULL);
	printf("CPU run time = %0.4f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	check_histogram(reference_histogram, num_elements, HISTOGRAM_SIZE); 
	
	/* Compute the histogram using pthreads. The result histogram should be stored in the 
     * histogram_using_pthreads array. */
	printf("Creating histogram using pthreads. \n");
	gettimeofday(&start,NULL);
	compute_using_pthreads(input_data, histogram_using_pthreads, num_elements, HISTOGRAM_SIZE);
	/* check_histogram(histogram_using_pthreads, num_elements, HISTOGRAM_SIZE); */
	gettimeofday(&stop,NULL);
	printf("CPU run time with pthreads = %0.4f s. \n",(float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	/* Compute the differences between the reference and pthread results. */
	diff = 0.0;
    for(i = 0; i < HISTOGRAM_SIZE; i++)
		diff = diff + abs(reference_histogram[i] - local_hist[0][i]);

	printf("Difference between the reference and pthread results: %f. \n", diff);
   
	/* cleanup memory. */
	free(input_data);
	free(reference_histogram);
	free(histogram_using_pthreads);

	pthread_exit(NULL);
}

/* This function computes the reference solution. */
void 
compute_gold(int *input_data, int *histogram, int num_elements, int histogram_size)
{
  int i;
  
   for(i = 0; i < histogram_size; i++)   /* Initialize histogram. */
       histogram[i] = 0; 

   for(i = 0; i < num_elements; i++)     /* Bin the elements. */
			 histogram[input_data[i]]++;
}


/* Write the function to compute the histogram using pthreads. */
void 
compute_using_pthreads(int *input_data, int *histogram, int num_elements, int histogram_size)
{
	int temp = num_elements % NUM_THREADS;
	int num_elements_per_thread =  (num_elements - temp)/NUM_THREADS;
	local_hist = (int **)malloc(sizeof(int *)*NUM_THREADS);
	for(int i=0;i<NUM_THREADS;i++) {
		local_hist[i]=(int *)malloc(sizeof(int)*HISTOGRAM_SIZE);
	}
	for(int i=0;i<NUM_THREADS;i++) {
		for(int j=0;j<HISTOGRAM_SIZE;j++) {
			local_hist[i][j]=0;
		}
	}
	pthread_barrier_init(&barr,NULL,NUM_THREADS);
	for(int i=0;i<NUM_THREADS;i++) {
		my_args = (myfun_args*)malloc(sizeof(myfun_args));
		my_args->thread_id = i;
		if (i==NUM_THREADS-1) {
			my_args->start_idx = num_elements_per_thread*i;
			my_args->end_idx = num_elements-1;
		}
		else {
		my_args->start_idx = num_elements_per_thread * i; 
		my_args->end_idx = (num_elements_per_thread *(i +1)) - 1;
		}	
		if((pthread_create(&my_thread[i],NULL,calc_hist,(void *)my_args))!=0) {
			printf("\nCannot create thread!");
			exit(0);
		}
	}
	pthread_join(my_thread[0],NULL);
}

void * calc_hist(void * my_args){
	myfun_args * my_args_final = (myfun_args *)my_args;
	int temp_id = my_args_final->thread_id;
	for(int i=my_args_final->start_idx;i<=my_args_final->end_idx;i++) {
		local_hist[temp_id][input_data[i]]++;
	}
	
	pthread_barrier_wait(&barr);

	int num_levels = log2(NUM_THREADS);
	int join_idx;
	for (join_idx=0;join_idx<num_levels;join_idx++){
	
	
		if((temp_id %(int)pow(2,join_idx+1))==0){
			int get_thread_id = temp_id + (int)pow(2,join_idx);
			pthread_join(my_thread[get_thread_id],NULL);		
			for(int i=0;i<HISTOGRAM_SIZE;i++) {
				local_hist[temp_id][i] += local_hist[get_thread_id][i];			
			}
		}
		else {
			pthread_exit(NULL);
		}
	}

}
		
/* Helper function to check for correctness of the resulting histogram. */
void 
check_histogram(int *histogram, int num_elements, int histogram_size)
{
	int sum = 0;
	for(int i = 0; i < histogram_size; i++)
		sum += histogram[i];

	printf("Number of histogram entries = %d. \n", sum);
	if(sum == num_elements)
		printf("Histogram generated successfully. \n");
	else
		printf("Error generating histogram. \n");
}



