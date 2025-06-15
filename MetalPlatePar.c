#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#define a 3.0 //Meters
#define Tbottom (273.15-20) //Kelvin
#define Ttop (273.15 + 200) //Kelvin
#define Tleft (273.15 + 150) //Kelvin
#define Tright (273.15 - 40) //Kelvin
#define TOLERANCE 0.01
#define maxIterations 10000000
#define NR_END 1
#define FREE_ARG char*


// Numerical Recipes.
void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
    fprintf(stderr,"Numerical Recipes run-time error...\n");
    fprintf(stderr,"%s\n",error_text);
    fprintf(stderr,"...now exiting to system...\n");
    exit(1);
}

double **dmatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
    long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
    double **m;
    /* allocate pointers to rows */
    m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
    if (!m) nrerror("allocation failure 1 in matrix()");
    m += NR_END;
    m -= nrl;
    /* allocate rows and set pointers to them */
    m[nrl]=(double *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
    if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
    m[nrl] += NR_END;
    m[nrl] -= ncl;
    for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
    /* return pointer to array of pointers to rows */
    return m;
}

void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by dmatrix() */
{
    free((FREE_ARG) (m[nrl]+ncl-NR_END));
    free((FREE_ARG) (m+nrl-NR_END));
}

// Main code.
int main(int argc, char **argv){
    int i,j,c; //Looping.
    double diff, maxDiff; //Stopping.
    int N=100; // Default.

    // Choosing N before running.
    for (i=1;i<argc;i++){
        if(argv[i][0]=='-'){
            switch(argv[i][1]){
                case 'N': sscanf(argv[i+1],"%d",&N);
                        break;
            }
        }
    }
    printf("N = %d\n", N); // After parsing N
    printf("Starting iterations...\n"); // Before the iteration loop

    double h = a/(N-1); 

    //Allocating Matrices.    
    double **T= dmatrix(0,N-1,0,N-1);
    double **T_prev= dmatrix(0,N-1,0,N-1); 




    // Boundary Conditions plus setting the interior = 0 for Jacobi Method.
    for (i=0;i<N;i++)
    {  
        for (j=0;j<N;j++)
        {
            if (j==0){
                T_prev[i][j] = Tbottom;
            }
            else if (j==N-1){
                T_prev[i][j] = Ttop;
            }
            else if (i==0){
                T_prev[i][j] = Tleft;
            }
            else if (i==N-1){
                T_prev[i][j] = Tright;
            }
            else{
                T_prev[i][j] = 0;
            }
        }
    }
//    double t_start = omp_get_wtime();
    for (c=0;c<maxIterations;c++){
        //Next approximation.
        #pragma omp parallel shared(T,T_prev) private(i,j) firstprivate(N) default(none)
        {
            #pragma omp for collapse(2)
            for (i=1;i<N-1;i++)
            {
                for (j=1;j<N-1;j++)
                {
                    T[i][j] = (T_prev[i-1][j] + T_prev[i+1][j] + T_prev[i][j-1] + T_prev[i][j+1])/4;
                }
            }
        }
        maxDiff=fabs(T[1][1]-T_prev[1][1]);
        //Updating the T array for next iterations.
        #pragma omp parallel for collapse(2) shared(T,T_prev) private(i,j,diff) firstprivate(N) reduction(max:maxDiff) default(none) 
        for (i=1;i<N-1;i++)
        {
            for (j=1;j<N-1;j++)
            {
                diff = fabs(T[i][j]-T_prev[i][j]);
                if(diff>maxDiff){
                    maxDiff=diff;
                }
                T_prev[i][j] = T[i][j];
            }
        }

        if (maxDiff<TOLERANCE){
            break;
        }
    }
 //   double t_end = omp_get_wtime();
 //   printf("After %d iterations with a time of %f", c,t_end-t_start);
    printf("Iterations: %d", c+1);
    //Deallocating.
    free_dmatrix(T,0,N-1,0,N-1);
    free_dmatrix(T_prev,0,N-1,0,N-1);
    return 0;
}