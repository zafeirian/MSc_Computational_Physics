#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define N 1000 // Grid Points.
#define L 2.0 // Length of rod.
#define a2 0.00011  // Thermal Diffusivity of Steel.
#define T0 673.15 // Kelvin (400 C).
//#define TL 253.15 // Kelvin (-20 C). 
#define TL 673.15
#define Ti 293.15 // Kelvin (20 C).
#define timestamps  500000 // 5000000
#define SaveTimeStamp 100 // Saving every 0.5 secs
#define TOLERANCE 1E-2

int main(void){
    omp_set_num_threads(1);

    clock_t start, end;
    double cpu_time_used;
    double h = L/(N-1); // Space step.
    double k = 0.01; // Time step. 
    double lambda = a2*k/(h*h);
        if (k > h*h/(2*a2) ){
            printf("You need to make your time step smaller.");
            return 0;
        }
    double T[N]; // T(x,t=stathero).
    double T_prev[N]; // T(x,t=stathero)
    int i,j;

    // Open file to save data.
    FILE *file = fopen("temperaturefinal1.txt", "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    T_prev[0] = T0; // Setting Boundary condition at x=0.
    for (i=1;i<N-1;i++) // Interior Points.
    {
        T_prev[i] = Ti;
    }
    T_prev[N-1] = TL;  //Setting Boundary Condition at x=L. 
    
    start = clock();

    // Calculating the Temp of the rod for all timesteps.
    for (j=1;j<timestamps;j++){
        //Calculating next time step.
        #pragma omp parallel shared(T,T_prev,lambda) private(i) default(none)
        {
            #pragma omp for schedule(static)
                for (i=0;i<N;i++) 
                {
                    if (i==0){
                        T[i] = T0;
                    }
                    else if (i==N-1){
                        T[i] = TL;   
                    }
                    else {
                        T[i] = (1-2*lambda)*T_prev[i] + lambda*(T_prev[i+1]+T_prev[i-1]);
                    }
                }   
        }

        // Saving selected timesteps on file.
        if (j%SaveTimeStamp==0)
        {
            for (i=0;i<N;i++){
                fprintf(file, "%.2lf ", T[i]);
            }
            fprintf(file, "\n");
        }

        // Check for uniform temperature.
        int is_uniform = 1;
        for (i = 1; i < N; i++) {
            // if T[i] - T0 is bigger than 0.01 then 
            if (fabs(T[i] - T0) > TOLERANCE) 
            {
                is_uniform = 0;
                break;
            }
        }

        // Making T_prev = T for the next loop.
        for (i=0;i<N;i++){
            T_prev[i] = T[i];
        }
    }
 
    end = clock();
    fclose(file);
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nTime taken: %lf",cpu_time_used);
    return 0;
}