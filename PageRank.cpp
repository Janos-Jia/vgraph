#include "vgraph.h"
#include <memory.h>
#include <xmmintrin.h>
#include <immintrin.h>

#define maxIter 5
#ifdef _TW10_
#define ALPHA (0.0004)
#else
#define ALPHA (0.002)
#endif

#define PR (0.85)
#define RESTPR (0.15)
#define CURRENTY (0.001)


float * pr;
int myprid;
float ** remote_pr;
int * remote_prid;

float * singlepr;
int mysingleprid;
float ** remote_singlepr;
int * remote_singleprid;

float * nextpr;
int mynextprid;
float ** remote_nextpr;
int * remote_nextprid;

int * out_degree;
float restnum;
int iteration;

int64_t row_begin,row_end;

int q1c,q2c;
int *q1;
int *q2;

void Pre_Compute()
{
    restnum = RESTPR/num_vertice;
#ifdef DISPRED
    int64_t llocal_vert = (int64_t)local_vert;
    Init_Shared_process<float>(SHARE_FILE1,llocal_vert,1,myprid,pr);
    Init_Shared_process<float>(SHARE_FILE1,llocal_vert,1+num_procs,mysingleprid,singlepr);
    Init_Shared_process<float>(SHARE_FILE1,llocal_vert,1+2*num_procs,mynextprid,nextpr);
    out_degree=(int *)_mm_malloc(local_vert*sizeof(int),64);
    q1=(int *)_mm_malloc(local_vert*sizeof(int),64);
    q2=(int *)_mm_malloc(local_vert*sizeof(int),64);
    MPI_Barrier(MPI_COMM_WORLD);
    int64_t all_local_vert[num_procs];
    
    MPI_Allgather(&llocal_vert,1 ,MPI_LONG,all_local_vert,1,MPI_LONG,MPI_COMM_WORLD);
   
    Attach_Shared_process<float>(SHARE_FILE1,all_local_vert,1,remote_prid,remote_pr);
    Attach_Shared_process<float>(SHARE_FILE1,all_local_vert,1+num_procs,remote_singleprid,remote_singlepr);
    Attach_Shared_process<float>(SHARE_FILE1,all_local_vert,1+2*num_procs,remote_nextprid,remote_nextpr);
#else
    Init_Shared_total<float>(SHARE_FILE1, num_vertice, 1, myprid, pr, 0);
    Init_Shared_total<float>(SHARE_FILE1, num_vertice, 2, mysingleprid, singlepr, 0);
    Init_Shared_total<float>(SHARE_FILE1, num_vertice, 3, mynextprid, nextpr, 0);
    out_degree = (int*)_mm_malloc(local_vert * sizeof(int), 64);
    q1 = (int*)_mm_malloc(local_vert * sizeof(int), 64);
    q2 = (int*)_mm_malloc(local_vert * sizeof(int), 64);
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    q1c=q2c=0;
#ifdef DISPRED
    for(IntV i=0;i<local_vert;i++)
    {
        out_degree[i]=rowstarts[i+1]-rowstarts[i];
        pr[i]=1.0;
        nextpr[i]=0.0;
        q1[i]=q2[i]=0;
    }

    for(IntV i=0;i<local_vert;i++)
    {
        if(out_degree[i])
        {
            singlepr[i] = pr[i]/out_degree[i];
            q1[q1c++]=i;
        }
        else
        {
            singlepr[i]=1.0;
            pr[i]=restnum;
            nextpr[i]=restnum;
        }
    }
#else
#ifdef OMP_
#pragma omp parallel for
#endif
    for (IntV i = myrowoffset; i < myrowend; i++)
    {
        out_degree[i - myrowoffset] = rowstarts[i + 1- myrowoffset] - rowstarts[i- myrowoffset];
        pr[i] = 1.0;
        nextpr[i] = 0.0;
        q1[i- myrowoffset] = q2[i- myrowoffset] = 0;
    }
#ifdef OMP_
#pragma omp parallel for
#endif
    for (IntV i = myrowoffset; i < myrowend; i++)
    {
        if (out_degree[i - myrowoffset])
        {
            singlepr[i] = pr[i] / out_degree[i - myrowoffset];
            q1[__sync_fetch_and_add(&q1c, 1)] = i;
        }
        else
        {
            singlepr[i] = 1.0;
            pr[i] = restnum;
            nextpr[i] = restnum;
        }
    }
#endif
    iteration=0;
    MPI_Barrier(MPI_COMM_WORLD);
}


void Compute()
{
    IntV temp_local;
    float temp_val;
    int64_t all_edgenum=0;
    int64_t total_alledge;
    int total_active;
    while (iteration < maxIter)
    {
        all_edgenum = 0;
        double mytime = MPI_Wtime();
#ifdef OMP_
#pragma omp parallel for
#endif
        for (IntV i = 0; i < q1c; i++)
        {
#ifdef DISPRED
            temp_local = q1[i];
#else
            temp_local = q1[i] - myrowoffset;
#endif
            row_begin = outrowstarts[temp_local];
            row_end = outrowstarts[temp_local + 1];
            all_edgenum += row_end - row_begin;
            for (int64_t j = row_begin; j < row_end; j++)
            {
#ifdef DCOL_
                nextpr[temp_local] += remote_singlepr[outcolumn[j].rk][outcolumn[j].id];
#else
#ifdef DISPRED
                nextpr[temp_local] += singlepr[outcolumn[j].id];
#else
                nextpr[q1[i]] += singlepr[outcolumn[j].id];
#endif
#endif
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double halftime = MPI_Wtime() - mytime;
        if (ranks == 0) printf("HalfTime : %f\n", halftime);
#ifdef OMP_
#pragma omp parallel for
        for (IntV i = 0; i < q1c; i++)
        {
            temp_local = q1[i];
            q2[i] = -1;
            temp_val = nextpr[temp_local] * PR + restnum;
            if (fabs(temp_val - pr[temp_local]) > CURRENTY)
            {
                nextpr[temp_local] = temp_val;
#ifdef DISPRED
                singlepr[temp_local] = nextpr[temp_local] / out_degree[temp_local];
#else
                singlepr[temp_local] = nextpr[temp_local] / out_degree[temp_local - myrowoffset];
#endif
                q2[i] = temp_local;
            }
        }

        for (IntV i = 0; i < q1c; i++)
        {
            if (q2[i] < 0)
            {
            }
            else q2[q2c++] = q2[i];
        }
#else
        for(IntV i=0;i<q1c;i++)
        {
            temp_local=q1[i];
            temp_val = nextpr[temp_local]*PR+restnum;
            if(fabs(temp_val-pr[temp_local])>CURRENTY)
            {
                nextpr[temp_local] = temp_val;
#ifdef DISPRED
                singlepr[temp_local]=nextpr[temp_local]/out_degree[temp_local];
#else
                singlepr[temp_local] = nextpr[temp_local] / out_degree[temp_local - myrowoffset];
#endif
                q2[__sync_fetch_and_add(&q2c,1)]=temp_local;
            }
        }
#endif
#ifdef DISPRED
        memset(pr,0,sizeof(float)*local_vert);
#else
        for (IntV i = myrowoffset; i < myrowend; i++) pr[i] = 0.0;
#endif

        MPI_Barrier(MPI_COMM_WORLD);
        mytime=MPI_Wtime()-mytime;
		float * tmp=pr;pr=nextpr;nextpr=tmp;
        int *tempint = q1; q1=q2;q2=tempint;

        q1c=q2c;
        q2c=0;
        iteration++;
        
        if(ranks==0) printf("Iteration : %d\n active_num : %d total_edge : %ld Time : %f\n\n",iteration, total_active, total_alledge,mytime);
    }
}

void Free_Compute()
{
#ifdef DISPRED
    Disattach_Shared_process<float>(myprid,pr,remote_prid,remote_pr);
    Disattach_Shared_process<float>(mysingleprid,singlepr,remote_singleprid,remote_singlepr);
    Disattach_Shared_process<float>(mynextprid,nextpr,remote_nextprid,remote_nextpr);
#else
    Disattach_Shared_total<float>(myprid, pr);
    Disattach_Shared_total<float>(mysingleprid, singlepr);
    Disattach_Shared_total<float>(mynextprid, nextpr);
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    _mm_free(out_degree);
    _mm_free(q1);
    _mm_free(q2);
    Free_Graph();
}

void Statistics_Print()
{
    return ;
}