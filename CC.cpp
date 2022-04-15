#include "vgraph.h"
#include <memory.h>
#include <xmmintrin.h>
#include <immintrin.h>

#define root 0
#ifdef _TW10_
#define ALPHA (0.0004)
#else
#define ALPHA (0.002)
#endif


#ifdef DISPRED
int localpredid;
int * local_pred;
int *predid;
int ** pred;
#else
int localpredid;
int * local_pred;
#endif

int myrunbufferid;
packed_edge * myrunbuffer;
int * runbufferid;
packed_edge ** runbuffer;

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
void Compute_Num_CC();;

void Pre_Compute()
{
#ifdef DISPRED
    Init_Shared_process<int>(SHARE_FILE1, local_vert, 1, localpredid, local_pred);
    MPI_Barrier(MPI_COMM_WORLD);
    IntV all_local_vert[num_procs];
#ifdef _INT64_
    MPI_Allgather(&local_vert, 1, MPI_LONG, all_local_vert, 1, MPI_LONG, MPI_COMM_WORLD);
#elif defined(_UNSIGNEDINT)
    MPI_Allgather(&local_vert, 1, MPI_UNSIGNED, all_local_vert, 1, MPI_UNSIGNED, MPI_COMM_WORLD);
#else
    MPI_Allgather(&local_vert, 1, MPI_INT, all_local_vert, 1, MPI_INT, MPI_COMM_WORLD);
#endif
    Attach_Shared_process<int>(SHARE_FILE1, all_local_vert, 1, predid, pred);
    MPI_Barrier(MPI_COMM_WORLD);
#else
    Init_Shared_total<int>(SHARE_FILE1, num_vertice, 1, localpredid, local_pred, 0);
#endif

#ifdef _PUSH_
    int64_t outproc[num_procs];
    remoterunoffset = (int64_t*)malloc(num_procs * sizeof(int64_t));
    for (int i = 0; i < num_procs; i++) outproc[i] = 0;

    for (int64_t i = 0; i < local_edge; i++)
    {
#ifdef DCOL_
        outproc[column[i].rk]++;
#else
        outproc[Owner(column[i].id)]++;
#endif
    }
    bufferoffset = (int64_t*)malloc(num_procs * sizeof(int64_t));
    bufindex = (int64_t*)malloc(num_procs * sizeof(int64_t));
    bufferoffset[0] = 0;
    for (int i = 0; i < num_procs - 1; i++) bufferoffset[i + 1] = bufferoffset[i] + outproc[i];
    int64_t tmp_to_proc[num_procs * num_procs];
    int64_t totalbuffer[num_procs];
    memset(totalbuffer, 0, num_procs * sizeof(int64_t));
    MPI_Allgather(outproc, num_procs, MPI_LONG, tmp_to_proc, num_procs, MPI_LONG, MPI_COMM_WORLD);
    for (int i = 0; i < num_procs; i++)
    {
        for (int j = i * num_procs; j < (i + 1) * num_procs; j++)
        {
            if (j % num_procs == ranks)
            {
                remoterunoffset[i] = totalbuffer[i];
            }
            totalbuffer[i] += tmp_to_proc[j];
        }
    }

    Init_Shared_process<packed_edge>(SHARE_FILE1, totalbuffer[ranks], 1 + num_procs, myrunbufferid, myrunbuffer);
    MPI_Barrier(MPI_COMM_WORLD);
    Attach_Shared_process<packed_edge>(SHARE_FILE1, totalbuffer, 1 + num_procs, runbufferid, runbuffer);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    q1 = (int*)_mm_malloc(local_vert * sizeof(int), 64);
    q2 = (int*)_mm_malloc(local_vert * sizeof(int), 64);
#ifdef _PULL_
    q3 = (int*)_mm_malloc(local_vert * sizeof(int), 64);
    q3c = 0;
#endif
    qc = q2c = 0;
    totalactive = nextactive = 0;
#ifdef DISPRED
    for (IntV i = 0; i < local_vert; i++) local_pred[i] = -1;
#else
    for (int64_t i = myrowoffset; i < myrowend; i++) local_pred[i] = -1;
#endif
    iteration = 0;

    warm_up();
}


void warm_up()
{
    qc = 0;
    if (Owner(root) == ranks)
    {
#ifdef DISPRED
        int local = Local_V(root);
#else
        int local = root;
#endif
        q1[qc++] = local;
        local_pred[local] = root;
}
    int64_t row_begin, row_end;
    int flag_b2u = 1;
    nextactive = 1;
    totalactive = 1;
    while (flag_b2u > 0)
    {
#if defined(_PUSH_) && defined(_PULL_)
        if (nextactive < ALPHA * num_vertice)
#endif
        {
#ifdef _PUSH_
            memcpy(bufindex, bufferoffset, num_procs * sizeof(int64_t));
            int64_t tmp_bufindex[num_procs * num_procs];

            for (int i = 0; i < qc; i++)
            {
                int localq = Local_V(q1[i]);
                row_begin = rowstarts[localq];
                row_end = rowstarts[localq + 1];
                for (int64_t j = row_begin; j < row_end; j++)
                {
#ifdef DCOL_
                    int to_where = column[j].rk;
                    myrunbuffer[bufindex[to_where]].v0 = Globalid(ranks, q1[i]);
                    myrunbuffer[bufindex[to_where]++].v1 = Globalid(to_where, column[j].id);
#else
                    int to_where = Owner(column[j].id);
                    myrunbuffer[bufindex[to_where]].v0 = q1[i] + myrowoffset;
                    myrunbuffer[bufindex[to_where]++].v1 = column[j].id;
#endif
                }
            }

            MPI_Allgather(bufindex, num_procs, MPI_LONG, tmp_bufindex, num_procs, MPI_LONG, MPI_COMM_WORLD);
            qc = 0;
            for (int i = 0; i < num_procs; i++)
            {
                int64_t end = tmp_bufindex[i * num_procs + ranks];
                for (int64_t j = remoterunoffset[i]; j < end; j++)
                {
#ifdef DISPRED
                    int local = Local_V(runbuffer[i][j].v1);
#else
                    int local = runbuffer[i][j].v1;
#endif
                    if (local_pred[local] < 0)
                    {
                        local_pred[local] = runbuffer[i][j].v0;
                        q1[qc++] = local;
                    }
                }
            }
            int64_t longqc = (int64_t)qc;
            MPI_Allreduce(&longqc, &nextactive, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
            totalactive += nextactive;
            if (nextactive == 0) flag_b2u = -1;
#endif

        }
#if defined(_PUSH_) && defined(_PULL_)
        else
#endif
        {
#ifdef _PULL_
            qc = 0;
#ifdef DISPRED
            for (IntV i = 0; i < local_vert; i++)
#else
            for (int64_t i = myrowoffset; i < myrowend; i++)
#endif
            {
                if (local_pred[i] == -1)
                {
#ifdef DISPRED
                    q1[qc++] = i;
#else
                    q1[qc++] = i - myrowoffset;
#endif
                }
            }

            while (1)
            {
                int q3c = 0;
                for (int64_t i = 0; i < qc; i++)
                {
                    int localq = q1[i];
                    row_begin = outrowstarts[localq];
                    row_end = outrowstarts[localq + 1];
                    for (int64_t j = row_begin; j < row_end; j++)
                    {
#ifdef DISPRED
                        int localc = outcolumn[j].id;
                        int ownerc = outcolumn[j].rk;
                        if (pred[ownerc][localc] > -1)
                        {
                            /*pred[q1[i]]=column[j];*/
                            q2[q3c] = q1[i];
                            q3[q3c++] = Globalid(outcolumn[j]);
                            break;
                        }
#else
#ifdef DCOL_
                        int global_temp = Globalid(outcolumn[j]);
#else
                        int global_temp = outcolumn[j].id;
#endif
                        if (local_pred[global_temp] > -1)
                        {
                            local_pred[localq + myrowoffset] = global_temp;
                            break;
                        }
#endif
        }

        }
                MPI_Barrier(MPI_COMM_WORLD);

                q2c = qc;
                qc = 0;
                for (int i = 0; i < q2c; i++)
                {
                    if (local_pred[q1[i] + myrowoffset] == -1)
                    {
                        q2[qc++] = q1[i];
                    }
                }

                q2c = q2c - qc;

                int* tmp = q1; q1 = q2; q2 = tmp;

                int64_t longq2c = (int64_t)q2c;
                MPI_Allreduce(&longq2c, &nextactive, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
                totalactive += nextactive;
                if (nextactive == 0)
                {
                    flag_b2u = -1;
                    break;
                }
                    }
#endif
                }
    }

#ifdef DISPRED
    for (IntV i = 0; i < local_vert; i++)
#else
    for (int64_t i = myrowoffset; i < myrowend; i++)
#endif
    {
        if (local_pred[i] == -1) local_pred[i]--;
        else local_pred[i] = -1;
    }
}


void Compute()
{
    qc = 0;
    if (Owner(root) == ranks)
    {
#ifdef DISPRED
        int local = Local_V(root);
#else
        int local = root;
#endif
        q1[qc++] = local;
        local_pred[local] = root;
    }
    int64_t row_begin, row_end;
    int flag_b2u = 1;
    nextactive = 1;
    totalactive = 1;
    if (ranks == 0) printf("[T2D] Iteration : %d\n  Current_Active : %ld\n  Total_Active : %ld\n\n", iteration++, nextactive, totalactive);
    while (flag_b2u > 0)
    {
#if defined(_PUSH_) && defined(_PULL_)
        if (nextactive < ALPHA * num_vertice)
#endif
        {
#ifdef _PUSH_
            double b2utime = MPI_Wtime();
            memcpy(bufindex, bufferoffset, num_procs * sizeof(int64_t));
            int64_t tmp_bufindex[num_procs * num_procs] __attribute__((aligned(64)));

            for (int i = 0; i < qc; i++)
            {
                int localq = q1[i] - myrowoffset;
                row_begin = rowstarts[localq];
                row_end = rowstarts[localq + 1];
                for (int64_t j = row_begin; j < row_end; j++)
                {
#ifdef DCOL_
                    int to_where = column[j].rk;
                    myrunbuffer[bufindex[to_where]].v0 = Globalid(ranks, q1[i]);
                    myrunbuffer[bufindex[to_where]++].v1 = Globalid(to_where, column[j].id);
#else
                    int to_where = Owner(column[j].id);
                    myrunbuffer[bufindex[to_where]].v0 = q1[i];
                    myrunbuffer[bufindex[to_where]++].v1 = column[j].id;
#endif
                }
            }

            MPI_Allgather(bufindex, num_procs, MPI_LONG, tmp_bufindex, num_procs, MPI_LONG, MPI_COMM_WORLD);
            qc = 0;
            for (int i = 0; i < num_procs; i++)
            {
                int64_t end = tmp_bufindex[i * num_procs + ranks];
                for (int64_t j = remoterunoffset[i]; j < end; j++)
                {
#ifdef DISPRED
                    int local = Local_V(runbuffer[i][j].v1);
#else
                    int local = runbuffer[i][j].v1;
#endif
                    if (local_pred[local] < 0)
                    {
                        local_pred[local] = runbuffer[i][j].v0;
                        q1[qc++] = local;
                    }
                }
            }
            int64_t longqc = (int64_t)qc;
            MPI_Allreduce(&longqc, &nextactive, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
            totalactive += nextactive;
            if (ranks == 0) printf("[T2D] Iteration : %d\n  Current_Active : %ld\n  Total_Active : %ld\n\n", iteration++, nextactive, totalactive);
            if (nextactive == 0) flag_b2u = -1;
            b2utime = MPI_Wtime() - b2utime;
            if (ranks == 0)	printf("[T2D] Iteration : %d Time : %f\n\n", iteration - 1, b2utime);
#endif

        }
#if defined(_PUSH_) && defined(_PULL_)
        else
#endif
        {
#ifdef _PULL_
            qc = 0;
#ifdef DISPRED
            for (IntV i = 0; i < local_vert; i++)
#else
            for (int64_t i = myrowoffset; i < myrowend; i++)
#endif
            {
                if (local_pred[i] == -1)
                {
                    q1[qc++] = i - myrowoffset;
                }
            }

            while (1)
            {
                double b2utime = MPI_Wtime();
                int q3c = 0;
                for (int64_t i = 0; i < qc; i++)
                {
                    int localq = q1[i];
                    row_begin = outrowstarts[localq];
                    row_end = outrowstarts[localq + 1];
                    for (int64_t j = row_begin; j < row_end; j++)
                    {
#ifdef DISPRED
                        int localc = outcolumn[j].id;
                        int ownerc = outcolumn[j].rk;
                        if (pred[ownerc][localc] > -1)
                        {
                            q2[q3c] = localq;
                            q3[q3c++] = Globalid(outcolumn[j]);
                            break;
                        }
#else
#ifdef DCOL_
                        int global_temp = Globalid(outcolumn[j]);
#else
                        int global_temp = outcolumn[j].id;
#endif
                        if (local_pred[global_temp] > -1)
                        {
                            local_pred[localq + myrowoffset] = global_temp;
                            break;
                        }
#endif
                    }

                }
                MPI_Barrier(MPI_COMM_WORLD);

               

                q2c = qc;
                qc = 0;
                for (int i = 0; i < q2c; i++)
                {
                    if (local_pred[q1[i] + myrowoffset] == -1)
                    {
                        q2[qc++] = q1[i];
                    }
                }

                q2c = q2c - qc;

                int* tmp = q1; q1 = q2; q2 = tmp;

                int64_t longq2c = (int64_t)q2c;
                MPI_Allreduce(&longq2c, &nextactive, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
                totalactive += nextactive;
                if (ranks == 0) printf("[B2U] Iteration : %d\n  Current_Active : %ld\n  Total_Active : %ld\n\n", iteration++, nextactive, totalactive);
                b2utime = MPI_Wtime() - b2utime;
                if (ranks == 0) printf("[B2U] Iteration : %d Time : %f\n\n", iteration - 1, b2utime);
                if (nextactive == 0)
                {
                    flag_b2u = -1;
                    break;
                }
            }
#endif
        }
    }
    qc = 0;
#ifdef DISPRED
    for (int64_t i = 0; i < local_vert; i++)
#else
    for (IntV i = myrowoffset; i < myrowend; i++)
#endif
    {
        if (local_pred[i] == -3) { local_pred[i] = i; }
        else if (local_pred[i] == -2)
        {
            local_pred[i] = i;
#ifdef DISPRED
            q1[qc++] = i;
#else
            q1[qc++] = i;
#endif
        }
        else { local_pred[i] = 0; }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int run_bfs_flag = 1;
    while (run_bfs_flag)
    {
        q2c = 0;
        for (int i = 0; i < qc; i++)
        {
#ifdef DISPRED
            row_begin = rowstarts[q1[i]];
            row_end = rowstarts[q1[i] + 1];
#else
            int localq = q1[i] - myrowoffset;
            row_begin = rowstarts[localq];
            row_end = rowstarts[localq + 1];
#endif
            for (int64_t j = row_begin; j < row_end; j++)
            {
#ifdef DISPRED
                int localc = Local_V(column[j]);
                int ownerc = Owner(column[j]);
                int tmppred = pred[ownerc][localc];
                if (tmppred < local_pred[q1[i]])
                {
                    local_pred[q1[i]] = tmppred;
                    q2c++;
                }
#else
                int tmppred = local_pred[column[j].id];
                if (tmppred < local_pred[q1[i]])
                {
                    local_pred[q1[i]] = tmppred;
                    q2c++;
                }
#endif
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allreduce(&q2c, &run_bfs_flag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
}


void Compute_Num_CC()
{
    int run_bfs_flag;
    for(int i=0;i<local_vert;i++)
    {
        rowstarts[i]=0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef DISPRED
    for(int64_t i=0;i<local_vert;i++)
    {
        int localc=Local_V(local_pred[i]);
        int ownerc=Owner(local_pred[i]);
        remote_rowstarts[ownerc][localc]=1;
    }
    #else
    for(int64_t i=myrowoffset;i<myrowend;i++)
    {
        int localc=Local_V(local_pred[i]);
        int ownerc=Owner(local_pred[i]);
        remote_rowstarts[ownerc][localc]=1;
    }
    #endif
    MPI_Barrier(MPI_COMM_WORLD);
    qc=0;
    for(int64_t i=0;i<local_vert;i++)
    {
        if(rowstarts[i]) qc++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&qc,&run_bfs_flag,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    if(ranks==0) printf("Total CC Num: %d \n\n",run_bfs_flag);
}


void Free_Compute()
{
    #ifdef DISPRED
    Disattach_Shared_process<int>(localpredid,local_pred,predid,pred);
    #else
    Disattach_Shared_total<int>(localpredid,local_pred);
    #endif
    Disattach_Shared_process<packed_edge>(myrunbufferid,myrunbuffer,runbufferid,runbuffer);
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
    Compute_Num_CC();
}