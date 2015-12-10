/* Gaussian elimination code.
 * Author: Harshvardhan Agrawal
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -std=c99 -O3 -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
 #include <xmmintrin.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

extern int compute_gold(float*, const float*, unsigned int);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
void gauss_eliminate_using_sse(const Matrix, Matrix);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, unsigned int, float);


int 
main(int argc, char** argv) {
    if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}	

    /* Allocate and initialize the matrices. */
	Matrix  A;                                              /* The N x N input matrix. */
	Matrix  U;                                              /* The upper triangular matrix to be computed. */
	
	srand(time(NULL));
		
    A  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1);      /* Create a random N x N matrix. */
	U  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);      /* Create a random N x 1 vector. */
		
	/* Gaussian elimination using the reference code. */
	Matrix reference;
	reference = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	struct timeval start, stop;	
	gettimeofday(&start, NULL);

	printf("Performing gaussian elimination using the reference code. \n");
	int status = compute_gold(reference.elements, A.elements, A.num_rows);

	gettimeofday(&stop, NULL);
	printf("CPU run time = %0.4f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	if(status == 0){
		printf("Failed to convert given matrix to upper triangular. Try again. Exiting. \n");
		exit(0);
	}
	status = perform_simple_check(reference); // Check that the principal diagonal elements are 1 
	if(status == 0){
		printf("The upper triangular matrix is incorrect. Exiting. \n");
		exit(0); 
	}
	printf("Gaussian elimination using the reference code was successful. \n");

	/* WRITE THIS CODE: Perform the Gaussian elimination using the SSE version. 
     * The resulting upper triangular matrix should be returned in U
     * */
	gettimeofday(&start, NULL);
	printf("Performing gaussian elimination using SSE. \n");
	gauss_eliminate_using_sse(A, U);

	gettimeofday(&stop, NULL);
	printf("CPU run time = %0.4f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));


	/* Check if the SSE result is equivalent to the expected solution. */
	int size = MATRIX_SIZE*MATRIX_SIZE;
	int res = check_results(reference.elements, U.elements, size, 0.001f);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	free(A.elements); A.elements = NULL;
	free(U.elements); U.elements = NULL;
	free(reference.elements); reference.elements = NULL;

	return 0;
}


void 
gauss_eliminate_using_sse(const Matrix A, Matrix U)                  /* Write code to perform gaussian elimination using OpenMP. */
{
	float *vector_a, *vector_b;
	void *allocation;
	int status;
	int i,j,k;
	__m128 *cp1 = (__m128 *) A.elements;
	__m128 *cp2 = (__m128 *) U.elements;
	int bound = (MATRIX_SIZE*MATRIX_SIZE)/4;
	__m128 m0_cp = _mm_set_ps1(0.0f);
	for (i = 0; i < bound ; i++){             /* Copy the contents of the A matrix into the U matrix. */
		*cp2 = _mm_add_ps(*cp1,m0_cp);
		*cp1++;
		*cp2++;
	}
	

	__m128 m1,m0_5,m0_6,m0_7;
	__m128 *src1,*src2;
	float temp,temp1;
	int loop_bound = MATRIX_SIZE/4;
	for(i=0;i<MATRIX_SIZE; i++){

		// Division segment
		src1 = (__m128 *) &U.elements[i*MATRIX_SIZE];
		temp = U.elements[i*MATRIX_SIZE + i];
		m0_5 = _mm_set_ps1(temp);
		
		for(j=0;j<loop_bound;j++){
			*src1 = _mm_div_ps(*src1,m0_5);
			src1++;
		}

		for(j=i+1;j<MATRIX_SIZE;j++){
			src1 = (__m128 *) &U.elements[i*MATRIX_SIZE];
			src2 = (__m128 *) &U.elements[j*MATRIX_SIZE];
			temp1 = (-1.0)*U.elements[j*MATRIX_SIZE + i];
			m0_6 = _mm_set_ps1(temp1);
//			m0_7 = _mm_set_ps1(0.0f);
			for(k=0;k<loop_bound;k++){
		//		m1 = _mm_add_ps(*src1,m0_7);
				m1 = _mm_mul_ps(*src1,m0_6);
				*src2 = _mm_add_ps(*src2,m1);
				src2++;
				src1++;
			}
		}
	}		
/*	for (j=0;j<MATRIX_SIZE*MATRIX_SIZE;j++){
		if(j%MATRIX_SIZE==0)
			printf("\n");
		printf("%f  ",U.elements[j]);
	}
*/	
}

		


int 
check_results(float *A, float *B, unsigned int size, float tolerance)   /* Check if refernce results match multi threaded results. */
{
	for(int i = 0; i < size; i++)
		if(fabsf(A[i] - B[i]) > tolerance)
			return 0;
	
    return 1;
}


/* Allocate a matrix of dimensions height*width. 
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization.
 * */
Matrix 
allocate_matrix(int num_rows, int num_columns, int init){
    Matrix M;
    int status;
    void *allocation;
    M.num_columns = M.pitch = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
/*    printf("\nBefore Status\n");
    status = posix_memalign(&allocation,16,sizeof(float)*MATRIX_SIZE);
    if(status != 0){
		printf("Error allocating aligned memory. \n");
		exit (0);
    }
    printf("\nAfter status\n");
    M.elements = (float *)allocation;
    printf("\nBefore getting values!\n");*/ 
    M.elements = (float *)malloc(sizeof(float)*size);
    for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }
    printf("\nAfter getting values\n");
    return M;
}	


float 
get_random_number(int min, int max){                                    /* Returns a random FP number between min and max values. */
	return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}

int 
perform_simple_check(const Matrix M){                                   /* Check for upper triangular matrix, that is, the principal diagonal elements are 1. */
    for(unsigned int i = 0; i < M.num_rows; i++)
        if((fabs(M.elements[M.num_rows*i + i] - 1.0)) > 0.001) return 0;
	
    return 1;
} 


