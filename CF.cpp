#include "vgraph.h"
#include <memory.h>
#include <xmmintrin.h>
#include <immintrin.h>

#define _INTVALUE_
#define WEIGHTS_

#define lambda 0.001
#define step 0.00000035
#define MaxIter 5
#define K 20


int latentid;
double* latent_curr;
double* error;


void warm_up();

void Pre_Compute()
{
    Init_Shared_total<double>(SHARE_FILE1, num_vertice*K, 1, latentid, latent_curr, 0);
    error = (double*)_mm_malloc(local_vert *K* sizeof(double), 64);
    int64_t Kmyrowend = myrowend * K;
#ifdef OMP_
#pragma omp parallel for
#endif
    for (int64_t i = myrowoffset * K; i < Kmyrowend; i++)
    {
        latent_curr[i] = 0.5;
    }

    int64_t Klocal_vert = local_vert * K;
#ifdef OMP_
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < Klocal_vert; i++)
    {
        error[i] = 0.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}



void Compute()
{
    
    for (int iter = 0; iter < MaxIter; iter++)
    {
        double b2utime = MPI_Wtime();
#ifdef OMP_
#pragma omp parallel for
#endif
        for (int i = 0; i < local_vert; i++)
        {
            int64_t row_begin = rowstarts[i];
            int64_t row_end = rowstarts[i+1];
            for (int64_t j = row_begin; j < row_end; j++)
            {
                double estimate = 0.0;
                int64_t myb = (myrowoffset + i) * K;
                int64_t mye = column[j].id * K;
                for (int k = 0; k < K; k++)
                {
                    estimate += latent_curr[myb + k] * latent_curr[mye + k];
                }
                double err = column[j].val-estimate;
                
                for (int k = 0; k < K; k++)
                {
                    error[i] += latent_curr[mye + k] * err;
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
#ifdef OMP_
#pragma omp parallel for
#endif
        for (int i = 0; i < local_vert; i++)
        {
            int64_t myb = (myrowoffset + i) * K;
            int64_t mye = i * K;
            for (int k = 0; k < K; k++)
            {
                latent_curr[myb + k] += step * (-lambda * latent_curr[myb + k] + error[ mye + k]);
                error[mye + k] = 0.0;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        b2utime = MPI_Wtime() - b2utime;
        if (ranks == 0) printf("[CF] Iteration : %d Time : %f\n\n", iter, b2utime);
    }
}


void Free_Compute()
{
    _mm_free(error);
    Disattach_Shared_total<double>(latentid, latent_curr);

    Free_Graph();
}

void Statistics_Print()
{
    return ;
}