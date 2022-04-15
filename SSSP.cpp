#include "vgraph.h"
#include <memory.h>
#include <xmmintrin.h>
#include <immintrin.h>

#define root 0
#define exampleV 
#ifdef _TW10_
#define ALPHA (0.0004)
#define BELTA (0.5)
#else
#define ALPHA (0.004)
#define BELTA (0.35)
#endif



#ifdef DISPRED
int localpredid;
Vtype* local_pred;
int *predid;
Vtype ** pred;
#else
int localpredid;
Vtype * local_pred;
#endif

int myrunbufferid;
halfpacked_edge * myrunbuffer;
int * runbufferid;
halfpacked_edge ** runbuffer;

int * q1;
int * q2;
int * q3;
int qc;
int q2c;
int q3c;
int iteration;

int64_t totalactive,nextactive;

int64_t * remoterunoffset;
int64_t * bufferoffset;
int64_t * bufindex;

void warm_up();

void Pre_Compute()
{
    #ifdef DISPRED
    int64_t nlocal_vert = (int64_t)local_vert;
    Init_Shared_process<Vtype>(SHARE_FILE1,nlocal_vert,1,localpredid,local_pred);
    MPI_Barrier(MPI_COMM_WORLD);
    int64_t all_local_vert[num_procs];
    
    MPI_Allgather(&nlocal_vert,1 ,MPI_LONG,all_local_vert,1,MPI_LONG,MPI_COMM_WORLD);
    Attach_Shared_process<Vtype>(SHARE_FILE1,all_local_vert,1,predid,pred);
    MPI_Barrier(MPI_COMM_WORLD);
    #else
    Init_Shared_total<Vtype>(SHARE_FILE1,num_vertice,1,localpredid,local_pred,0);
    #endif
    
#ifdef _PUSH_
    int64_t outproc[num_procs];
    remoterunoffset = (int64_t *)malloc(num_procs*sizeof(int64_t));
    for(int i=0;i<num_procs;i++) outproc[i]=0;

    for(int64_t i=0;i<local_edge;i++)
    {
#ifdef DCOL_
        outproc[column[i].rk]++;
#else
        outproc[Owner(column[i].id)]++;
#endif
    }
    bufferoffset = (int64_t *)malloc(num_procs*sizeof(int64_t));
    bufindex = (int64_t *)malloc(num_procs*sizeof(int64_t));
    bufferoffset[0]=0;
    for(int i=0;i<num_procs-1;i++) bufferoffset[i+1]=bufferoffset[i]+outproc[i];
    int64_t tmp_to_proc[num_procs*num_procs];
    int64_t totalbuffer[num_procs];
    memset(totalbuffer,0,num_procs*sizeof(int64_t));
    MPI_Allgather(outproc,num_procs,MPI_LONG,tmp_to_proc,num_procs,MPI_LONG,MPI_COMM_WORLD);
	for(int i=0;i<num_procs;i++)
	{
		for(int j=i*num_procs;j<(i+1)*num_procs;j++)
		{
			if(j%num_procs==ranks)
			{
				remoterunoffset[i]=totalbuffer[i];
			}
			totalbuffer[i]+=tmp_to_proc[j];
		}
    }

    Init_Shared_process<halfpacked_edge>(SHARE_FILE1,totalbuffer[ranks],1+num_procs,myrunbufferid,myrunbuffer);
    MPI_Barrier(MPI_COMM_WORLD);
    Attach_Shared_process<halfpacked_edge>(SHARE_FILE1,totalbuffer,1+num_procs,runbufferid,runbuffer);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    q1=(int *)_mm_malloc(local_vert*sizeof(int),64);
    q2=(int *)_mm_malloc(local_vert*sizeof(int),64);
    #ifdef _PULL_
    q3 = (int *)_mm_malloc(local_vert*sizeof(int),64);
    q3c=0;
    #endif
    qc=q2c=0;
    totalactive=nextactive=0;
    #ifdef DISPRED
    for(IntV i=0;i<local_vert;i++) local_pred[i]= 10000.0;
    #else
    for(int64_t i=myrowoffset;i<myrowend;i++) local_pred[i]=10000.0;
    #endif
    iteration=0;

}



void Compute()
{
    int change_flag = 1;
    float temp_dist;
    int write_flag;
    qc=0;
    if(Owner(root)==ranks)
    {
        #ifdef DISPRED
        int local=Local_V(root);
        #else
        int local=root;
        #endif
        q1[qc++]=local - myrowoffset;
        local_pred[local]=0.0;
    }
    int64_t row_begin,row_end;
    int flag_b2u=1;
    nextactive=1;
    totalactive=1;
    if(ranks==0) printf("[T2D] Iteration : %d\n  Current_Active : %ld\n  Total_Active : %ld\n\n",iteration++,nextactive,totalactive);
    while(flag_b2u>0)
    {
        #if defined(_PUSH_) && defined(_PULL_)
        if(change_flag)
        #endif
        {
            #ifdef _PUSH_
            double b2utime=MPI_Wtime();
            memcpy(bufindex,bufferoffset,num_procs*sizeof(int64_t));
            int64_t tmp_bufindex[num_procs*num_procs] __attribute__((aligned(64)));

            for(int i=0;i<qc;i++)
            {
                int localq = q1[i];
                row_begin = rowstarts[localq];
                row_end = rowstarts[localq+1];
                for(int64_t j=row_begin;j<row_end;j++)
                {
#ifdef DISPRED
                    temp_dist = local_pred[localq] + column[j].val;
                    if (pred[column[j].rk][column[j].id] > temp_dist)
                    {
                        cas(&pred[column[j].rk][column[j].id], pred[column[j].rk][column[j].id], temp_dist);
                        myrunbuffer[bufindex[column[j].rk]].v0 = column[j].id;
                        myrunbuffer[bufindex[column[j].rk]++].val = temp_dist;
                    }
#else
                    temp_dist = local_pred[q1[i] + myrowoffset] + column[j].val;
                    if (local_pred[column[j].id] > temp_dist)
                    {
                        cas(&local_pred[column[j].id], local_pred[column[j].id], temp_dist);
                        int to_where = Owner(column[j].id);
                        myrunbuffer[bufindex[to_where]].v0 = column[j].id;
                        myrunbuffer[bufindex[to_where]++].val = temp_dist;
                    }
#endif
                }
            }
            
            MPI_Allgather(bufindex,num_procs,MPI_LONG,tmp_bufindex,num_procs,MPI_LONG,MPI_COMM_WORLD);
            qc=0;
            for(int i=0;i<num_procs;i++)
            {
                int64_t end = tmp_bufindex[i*num_procs+ranks];
                for(int64_t j=remoterunoffset[i];j<end;j++)
                {
                    #ifdef DISPRED
                    int local = runbuffer[i][j].v0;
                    if(local_pred[local]< runbuffer[i][j].val)
                    {
                        
                    }
                    else q1[qc++] = local;
                    #else
                    int local = runbuffer[i][j].v0;
                    if (local_pred[local] < runbuffer[i][j].val)
                    {

                    }
                    else q1[qc++] = local-myrowoffset;
                    #endif
                }
            }
            int64_t longqc = (int64_t)qc;
            MPI_Allreduce(&longqc,&nextactive,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
            totalactive+=nextactive;
            if(ranks==0) printf("[T2D] Iteration : %d\n  Current_Active : %ld\n  Total_Active : %ld\n\n",iteration++,nextactive,totalactive);
            if(nextactive==0) flag_b2u=-1;
            b2utime = MPI_Wtime()-b2utime;
			if(ranks==0)	printf("[T2D] Iteration : %d Time : %f\n\n",iteration-1,b2utime);
            if (change_flag == 1)
            {
                if (nextactive < ALPHA * num_vertice)
                {
                    change_flag = 0;
                }
            }
            #endif

        }
        #if defined(_PUSH_) && defined(_PULL_)
        else
        #endif
        {
            #ifdef _PULL_
            double b2utime = MPI_Wtime();
            qc = 0;
#ifdef DISPRED
            for (IntV i = 0; i < local_vert; i++)
            {
                write_flag = 0;
                row_begin = outrowstarts[i];
                row_end= outrowstarts[i+1];
                for (int j = row_begin; j < row_end; j++)
                {
                    temp_dist = pred[column[j].rk][column[j].id] + column[j].val;
                    if (local_pred[i] > temp_dist)
                    {
                        local_pred[i] = temp_dist;
                        write_flag = 1;
                    }
                }
                if (write_flag) q1[qc++] = i;
            }
#else
            for (int64_t i = myrowoffset; i < myrowend; i++)
            {
                write_flag = 0;
                row_begin = outrowstarts[i- myrowoffset];
                row_end = outrowstarts[i + 1 - myrowoffset];
                for (int j = row_begin; j < row_end; j++)
                {
                    temp_dist = local_pred[column[j].id] + column[j].val;
                    if (local_pred[i] > temp_dist)
                    {
                        local_pred[i] = temp_dist;
                        write_flag = 1;
                    }
                }
                if (write_flag) q1[qc++] = i - myrowoffset;
            }
#endif
            int64_t longqc = (int64_t)qc;
            MPI_Allreduce(&longqc, &nextactive, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
            totalactive += nextactive;
            if (ranks == 0) printf("[B2U] Iteration : %d\n  Current_Active : %ld\n  Total_Active : %ld\n\n", iteration++, nextactive, totalactive);
            if (nextactive == 0) flag_b2u = -1;
            b2utime = MPI_Wtime() - b2utime;
            if (ranks == 0)	printf("[B2U] Iteration : %d Time : %f\n\n", iteration - 1, b2utime);
            if(nextactive < BELTA * num_vertice) change_flag = 2;
            #endif
        }
    }
}


void Free_Compute()
{
    #ifdef DISPRED
    Disattach_Shared_process<Vtype>(localpredid,local_pred,predid,pred);
    #else
    Disattach_Shared_total<Vtype>(localpredid,local_pred);
    #endif
    Disattach_Shared_process<halfpacked_edge>(myrunbufferid,myrunbuffer,runbufferid,runbuffer);
    MPI_Barrier(MPI_COMM_WORLD);
    free(remoterunoffset);
	_mm_free(q1);
	_mm_free(q2);
    #ifdef _PULL_
    _mm_free(q3);
    #endif
    free(bufferoffset);
    free(bufindex);
    Free_Graph();
}

void Statistics_Print()
{
    return ;
}