#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/sysinfo.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int stride = 50;
int nThreads = 8;

void print_matrix(double * A , int size);

static double get_wall_seconds() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000;
  return seconds;
}

void print_matrix(double * A , int size){

	for (int i = 0; i < size; ++i){
		for (int j = 0; j < size; ++j){
			printf("%0.0f ",  (A[i * size + j] ) );
		} printf("\n");
	}

// 		for (int j = 0; j < size; ++j){
// 			printf("%0.0f ",  (A[j * size + j] ) );
// 		} printf("\n");

// printf("\n");
}

double* multiply_matrix(double * A, double * B, int size){

	double * C  = malloc(size * size * sizeof(double *));
	#pragma omp parallel for  schedule(dynamic,stride)  num_threads(nThreads)
	for (int i = 0; i < size; i++) 
        for (int k = 0; k < size; k++) 
            for (int j = 0; j < size; j++)
                C[size *i + j] += A[size*k + j] * B[size*i + k];
return C;
}

double * transpose(double * A, int size){

	double * At  = malloc(size * size * sizeof(double *));
	#pragma omp parallel for  schedule(dynamic,stride)  num_threads(nThreads)	
		for (int i = 0; i < size; ++i){
			for (int j = 0; j < size; ++j){
				At[i*size + j] = A[j*size + i] ;
			}
		}
		return At;
}

double * incomplete_cholesky(double *A,  int n) {

	double * L  = malloc(n * n * sizeof(double *));
	for (int i = 0; i < n; ++i){
		for (int j = 0; j < n; ++j){
			L[i*n + j] = 0;
		}
	}

    for (int i = 0; i < n; i++)
        for (int j = 0; j < (i+1); j++) {
        	if (A[i * n + j] != 0) {
	        	
	            double s = 0;
	            for (int k = 0; k < j; k++)
	                s += L[i * n + k] * L[j * n + k];
	            if (i == j){
	            	  L[i * n + j] = sqrt(A[i * n + i] - s) ;
	            }
	            else {
	            	 L[i * n + j] = (1.0 / L[j * n + j] * (A[i * n + j] - s));
	            }
        }
    }
    return L;
}

void incomplete_LDL(double *A, double *D, double * L, int n) {

    for (int j = 0; j < n; j++){
    	double s = 0;
    	for (int k = 0; k < j; k++){
    		s += L[j * n + k] * L[j * n + k] * D[k * n + k];
    	} 
    	D[j * n + j] = A[j * n + j] - s;

        for (int i = j; i < n ; i++) {
     
        	if (A[i * n + j] != 0) {    	
	            s = 0;
	            for (int k = 0; k < j; k++)
	                s += L[i * n + k] * L[j * n + k] * D[k * n + k] ;
	            L[i * n + j] = (1.0 / D[j * n + j] * (A[i * n + j] - s));
        	}	
    	} 
	}
}

double * diag_matrix( int size){

	double * S  = malloc(size * size * sizeof(double *)); 	
	for (int i = 0; i < size; ++i){
		for (int j = 0; j < size; ++j){
			S[i*size + j] = 0;
		 }
	S[i*size + i] = 1; //(A[i*size + i]);
	}
	return S;
}

void forward_substitution(double * L, double * b , double *x, int size){

	for (int i = 0; i < size; ++i){
		
		double s = 0;
		for (int j = 0; j < i; ++j){
			s += L[i*size + j ]* x[j];
		}
		x[i] = (b[i] - s) /L[i*size + i ];
	}
}

void invert(double * D, int size){
	for (int i = 0; i < size; ++i)
	{
		D[i*size + i ] = 1 / D[i*size + i ];
	}
}

double * invert_cholesky(double * A, int size){

 	double* L =	incomplete_cholesky(A, size);
 	double *S = diag_matrix( size);
    double *X = malloc(size * size * sizeof(double *));
	for (int i = 0; i < size; ++i){
	 	forward_substitution(L, (S + i*size) , (X + i*size), size);
	}	
	double * Xt = transpose(X, size);
	double *R = multiply_matrix(Xt,X,size);
	R = multiply_matrix(R,A,size);	

	free(L);
	free(S);
	free(X);
	free(Xt);	

	return R;
}

double * invert_LDL(double * A, int size){

	double *L = malloc(size * size * sizeof(double *));
	double *D = malloc(size * size * sizeof(double *));
	incomplete_LDL(A,D,L,size);

	double *S = diag_matrix(size);
    double *X = malloc(size * size * sizeof(double *));
	for (int i = 0; i < size; ++i){
	 	forward_substitution(L, (S + i*size) , (X + i*size), size);
	}

	invert(D, size);
	double * Xt = transpose(X, size);
	
	double *XtD = multiply_matrix(Xt,D,size);
	double *R = multiply_matrix(XtD,X,size);

	R= multiply_matrix(R,A,size);	

	free(L);
	free(D);
	free(S);
	free(X);
	free(Xt);
	free(XtD);

	return R;
}


int main(int argc, char **argv) {

	int size =  atoi(argv[1]);

	 double *matrix = malloc(size * size * sizeof(double *));

	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++) { 
			matrix[ i* size + j ] = 0;
		  	if (  abs(i - j ) == 1 ){
	 		  	matrix[ i* size + j ] = -1;
	 	  } if(i==j){
	 	  	    matrix[ i* size + j ] = 2;
			}
		  }
		  printf("%f \n", sin(i) );
		}
	//double matrix[9] = {4, 12, -16, 12, 37, -43, -16, -43, 98};

	double *M1 = invert_cholesky(matrix, size);

	double *M2 = invert_LDL(matrix, size);

	free(matrix);
	free(M1);
	free(M2);
}

