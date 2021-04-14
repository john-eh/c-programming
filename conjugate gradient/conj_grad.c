#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <time.h>
#include <sys/sysinfo.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef BLAS

void dcopy_(const int *N, const double *DX, const int *INCX, 
		const double *DY,const int *INCY);

double ddot_(	const int *N, const double *DX, const int *INCX,
		const double *DY, const int *INCY );
	
void daxpy_(const int *N, const double *DA, const double *DX,
		const int *INCX, const double *DY, const int *INCY);	

void daxpby_(const int *N, const double *DA, const double *DX,
		const int *INCX,  const double *DB, const double *DY, 
		const int *INCY);

void dgemm_(const char *ta,const char *tb,
	    const int *n, const int *k, const int *l,
	    const double *alpha,const double *A,const int *lda,
	    const double *B, const int *ldb,
	    const double *beta, double *C, const int *ldc);

void dgemv_(const char *TRANS, const int *M, const int *N, 
		const double *ALPHA, const double *A, const int *LDA, 
		const double  *X, const int *INCX, const double *BETA,
		const double *Y, const int 	*INCY);

#endif

int stride = 50;

static double get_wall_seconds() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000;
  return seconds;
}

double vecVecMultiplication(double* vecA, double* vecB, int size, int threadNo, int avx_enable) {
	/* Computes dot product of two vectors */
	double sum = 0;
	int i;

	#pragma omp parallel num_threads(threadNo)
  	{
  		if (avx_enable){ 	
  		#pragma  omp for schedule(dynamic,stride) reduction(+: sum)
			for (i = 0; i < size; i+= 4) {

				/* Vector instructions to multiply and decrease two vectors down to one value */
				__m256d va  = _mm256_loadu_pd(  (vecA + i) );
				__m256d vb  = _mm256_loadu_pd(  (vecB + i) );		
				__m256d add = _mm256_mul_pd (va,  vb);

				__m256d t1 = _mm256_hadd_pd(add,add);
				__m128d t3 = _mm256_extractf128_pd(t1,1);
				__m128d t4 = _mm_add_sd(_mm256_castpd256_pd128(t1),t3);
				    
				sum += _mm_cvtsd_f64(t4);   
	    	} 
	   } else {
	    #pragma  omp for schedule(dynamic,stride) reduction(+: sum)
	    	for (i = 0; i < size; i++) {
	    		sum += (vecA[i] * vecB[i]);   
	    	}
  	}

  }
	return sum;
}

void vecScalarMultiplication(double * vecA, double * vecB, double * result, double scalar, int size, int nThreads, int avx_enable ){
 /* adds vector to other with scalar as:
    result = vecA + scalar*vecB */

	int i;
	if (avx_enable){
		const __m256d scalarv = _mm256_set1_pd ( scalar );
		#pragma omp parallel for schedule(dynamic,stride) num_threads(nThreads) shared(scalarv)
			for (i = 0; i < size; i+= 4) {

					__m256d va =  _mm256_loadu_pd(  (vecA + i) );
					__m256d vb =  _mm256_loadu_pd(  (vecB + i) );	
					__m256d	v2 =  _mm256_mul_pd(vb, scalarv);
					__m256d	v3 =  _mm256_add_pd(va, v2 );

					_mm256_storeu_pd(  (result +i), v3 );

			}
	} else {
	#pragma omp parallel for schedule(dynamic,stride) num_threads(nThreads)
		for (i = 0; i < size; i++) {
			result[i] = vecA[i] + scalar * vecB[i];
		}
	}
}

void vecMatMultiplication(double* matrix, double* p, double * result, int size, int nThreads, int avx_enable) {
	/* Vector-matrix multiplication, with resulting array stored in result */

	int i, j;
	#pragma omp parallel for  schedule(dynamic,stride)  num_threads(nThreads)
		for (i = 0; i < size; i++) {
			result[i] = 0.0;
			if (avx_enable){
				for (j = 0; j < size; j+=4 ) {

					/* Vector instructions to multiply and decrease two vectors down to one value */
					__m256d va =  _mm256_loadu_pd( (matrix+ i*size + j) );
					__m256d vb =  _mm256_loadu_pd(  (p+j) );		
					__m256d vadd = _mm256_mul_pd (va,  vb);
		
					__m256d v1 = _mm256_hadd_pd(vadd,vadd);
				    __m128d v2 = _mm256_extractf128_pd(v1,1);
				    __m128d v3 = _mm_add_sd(_mm256_castpd256_pd128(v1),v2);
				    
				    result[i] += _mm_cvtsd_f64(v3);   				
				}
			} else {
				result[i] = 0.0;
				for (j = 0; j < size; j++ ) {
			    	result[i] += matrix[i*size + j]* p[j];   				
				}			
			}
		}
}


int read_from_file(double * matrix, double *b, int size, const char * title){

	FILE * fp;
	fp = fopen(title, "r+");
	int read_int;

    read_int = fread(matrix, sizeof(double), size, fp);
		if (read_int < 0){
			return -1;
		}

	read_int = fread(b, sizeof(double), size, fp);
	if (read_int < 0){
		return -1;
	}

	fclose(fp);
	return 0;
}

void save_to_file( double * xvector, int size, int debug){
	
	FILE * fp;
	if (debug){
		fp = fopen("results.txt", "w+");
		for (int i = 0; i < size; ++i){
  			fprintf(fp, "%f \n",xvector[i] );
 		 } 
  } else {
		fp = fopen("results", "w+");
   	 	fwrite(xvector, sizeof(double), size, fp);
	}
  	fclose(fp);
}

void init_preconditioner(double * A, double * M, int size ){

	for (int i = 0; i < size; ++i){
		M[i] = 1.0/A[i*size + i];
	}
}

int main(int argc, char **argv) {
  
  	int      i, k, size, max_iterations, nThreads;
  	double	 val_a, val_b, tolerance;

  	if (argc < 2){
  		printf("Function to solve Ax = b using conjugate gradiant\n");
  		printf("\n");
  		printf("Arguments are: \n");
  		printf("(obligatory) n: size of square matrix\n");
  		printf("(optional) threadNo: number of threads to run \n");
  		printf("(optional) avx_enable: 1 if AVX instructions are toggled on, 0 if toggled off \n" );
  		printf("(optional) tolerance: maximum error tolerated  \n");
  		printf("(optional) iterations: maximum number of iterations  \n");
  		printf("(optional) file_name: file containing n+1 lines; n lines for square positive definite nxn matrix and 1 line for 1xn vector  \n");
  		return -1;
  	}

  	size = atoi(argv[1]);/* Determines dimension of matrix A */

	const char * fileName;
	int  	avx_enable 		= 1;
	int generate_matrix 	= 1;
	tolerance 				= 0.0001;
	max_iterations 			= 5000;

   	/* If number of threads not specified, get highest divisible by size */
   	val_a = size; val_b = get_nprocs();
  	while ( ceil(val_a/val_b) != floor(val_a/val_b)		) {
  	  	val_b =val_b/2;
  	}  
	nThreads = val_b;

	if( argc > 2){
		nThreads = atoi(argv[2]);
  } if( argc > 3){
  		avx_enable = atoi(argv[3]);
  } if( argc > 4){
  		tolerance = atoi(argv[4]);
  } if (argc > 5){
  		max_iterations = atoi(argv[5]);
  } if (argc > 6){
  		fileName = argv[6];
  } 

	printf("%d\n", nThreads);


	double *matrix = malloc(size * size * sizeof(double *));

	double* b = malloc(size * sizeof(double));

	/* If a matrix is not provided, it will be generated */
	if (generate_matrix == 1){

		// for (i = 0; i < size; i++) { 
		// 	for (int j = 0; j < size; j++) { 
		//   	  	matrix[i][j] = 0;
		//   		if(i==j){
		//           matrix[i][j] = i+1;
		// 	    }
		//     }
		//   }
		for (int i = 0; i < size; i++){
			for (int j = 0; j < size; j++) { 
				matrix[ i* size + j ] = 0;
		  		if (  abs(i - j ) == 1 ){
		  		  	matrix[ i* size + j ] = -1;
		  	  } if(i==j){
		        	matrix[ i* size + j ] = 2;
			   		}
		  }
		}
		// for (int i = 0; i < size; ++i){
		// 	for (int j = 0; j < size; ++j){
		// 		if (i == j){
		// 			matrix[i][j] = 50;
		// 		}
		// 		else if (abs (i-j) < 4)
		// 		{
		// 			matrix[i][j] = -10/abs(i-j);
		// 		}
		// 	}
		// }


		matrix[0] = 1000;
		matrix[size*(size-1) + size-1] = 1000;
		for (i = 0; i < size; i++) {
			//b[i] =1 + i%10 ;  //sin(i);
			b[i] = 10 + 5*sin(i);
		}
	} else{
		int read_int = read_from_file(matrix, b, size, fileName);
		if (read_int == -1){
			printf("Error, couldn't read file \n");
			return -1;
		}
	}

	double beta;
	//double alpha;
	double* x = malloc(size * sizeof(double));
	double* Ap = malloc(size * sizeof(double));
	double* x_old = malloc(size * sizeof(double));
	double* p = malloc(size * sizeof(double));
	double* p_old = malloc(size * sizeof(double));
	double* r = malloc(size * sizeof(double));
	double* r_old = malloc(size * sizeof(double));


	double tic =  get_wall_seconds();
	  
	/* With a starting guess of x = 0 we get starting value for r as r = b */
	memset(x,0,size);
	memcpy(r, b, (size * sizeof(double)));
	memcpy(p, r, (size * sizeof(double)));
	memcpy(r_old, b, (size * sizeof(double)));

	double test; 

#ifdef BLAS

	k = 0;
	while ((k < max_iterations)  ) {
		k++;

   		int incr = 1;
		dcopy_(&size, r, &incr, r_old, &incr);
		dcopy_(&size, p, &incr, p_old, &incr);
		dcopy_(&size, x, &incr, x_old, &incr);

   		double d1, d2;
		char trans = 'N';
		double alpha = 1;
		beta = 0;
		int lda  = size;
		int incx = 1;
		dgemv_(&trans, &size, &size, &alpha, matrix, &lda, p, &incx, &beta, Ap, &incx);  		

		d1 = ddot_(	&size, r, &incx, r, &incx);
		d2 = ddot_(	&size, p, &incx, Ap, &incx);

	 	double alpha1 = d1/d2;
	 	double alpha2 = -alpha1;

	 	daxpy_(&size, &alpha1, p, &incx, x, &incx);
	 	daxpy_(&size, &alpha2, Ap, &incx, r, &incx);
			  	  		  
	    test = ddot_(	&size, r, &incx, r, &incx);
	    if (test < tolerance){
	    	break;
	    }

	    double c1, c2;

	    c1 = ddot_(	&size, r, &incx, r, &incx);
	    c2 = ddot_(	&size, r_old, &incx, r_old, &incx);


		beta = c1/c2;

		double alp = 1;
		daxpby_(&size, &alp, r, &incx, &beta, p, &incx);

	}
	double toc =  get_wall_seconds();	
	printf("test blas\n");
#else

	k = 0;
	while ((k < max_iterations)  ) {
		k++;

		/* Updating values of array */
		memcpy(r_old, r, (size * sizeof(double)));
		memcpy(p_old, p, (size * sizeof(double)));
		memcpy(x_old, x, (size * sizeof(double)));

   		double d1, d2;
   		/* Matrix computation for computing alpha */
		vecMatMultiplication( matrix, p, Ap, size, nThreads, avx_enable);


  		d1 = vecVecMultiplication(r, r, size, nThreads, avx_enable);
  		d2 = vecVecMultiplication(p, Ap, size, nThreads, avx_enable);

	 	double alpha = d1/d2;

	 	/* Updates x_k+1 and r_k+1 */
	 	vecScalarMultiplication(x_old, p, x, alpha, size, nThreads, avx_enable);
	 	vecScalarMultiplication(r_old, Ap, r, -alpha, size, nThreads, avx_enable);

		/* If residue vector r is now small enough we exit the loop */
	    test = vecVecMultiplication(r, r, size, nThreads, avx_enable);

	    if (test < tolerance){
	    	break;
	    }

	    double c1, c2;
	    c1 = vecVecMultiplication(r, r, size, nThreads, avx_enable);
	    c2 = vecVecMultiplication(r_old, r_old, size, nThreads, avx_enable);

		beta = c1/c2;

		/* p is updated with beta */
		vecScalarMultiplication(r, p_old, p, beta, size, nThreads, avx_enable);

	}
	double toc =  get_wall_seconds();	
	printf("test inte-blas\n");

#endif	
	printf("time 1: %f \n",toc - tic );
	printf("Number of iterations: %d \n", k);
	printf("step: %d, Remainder: %f\n", k ,test); 

	double sum = 0;
	vecMatMultiplication( matrix, x, Ap, size, nThreads, avx_enable);
	for (int i = 0; i < size; ++i){
		sum += (Ap[i] - b[i]);
	}	printf("sum of Ax-b: %f \n",sum );


	int debug = 0;
	save_to_file(x, size, debug); 

	free((void *)matrix);

	free(b);
	free(x);
  	free(Ap);
	free(p);
	free(r);
	free(x_old);
	free(p_old);
	free(r_old);
  
  return 0;
}
