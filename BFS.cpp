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



__m512i vindex,myoffset,LoadData;
__m512i vindex1,vindex2,vindex3,vindex4,vindex5,vindex6,vindex7;
__m512i LoadData1,LoadData2,LoadData3,LoadData4,LoadData5,LoadData6,LoadData7;
__m256i loadData,storeData;
__m256i loadData1,loadData2,loadData3,loadData4,loadData5,loadData6,loadData7;
__m256i storeData1,storeData2,storeData3,storeData4,storeData5,storeData6,storeData7;
__mmask8 mk8;
void imm_vector(const int * q2, const int * q3, int * & local_pred, int q3c)
{
    int nBlockWidth = 64;
    int cntBlock = q3c / nBlockWidth;
    int cntRem = q3c % nBlockWidth; 
    for(int i=0;i<cntBlock;i++)
    {
        storeData = _mm256_mask_load_epi32(storeData,mk8,q3);
        storeData1 = _mm256_mask_load_epi32(storeData,mk8,q3+8);
        storeData2 = _mm256_mask_load_epi32(storeData,mk8,q3+16);
        storeData3 = _mm256_mask_load_epi32(storeData,mk8,q3+24);
        storeData4 = _mm256_mask_load_epi32(storeData,mk8,q3+32);
        storeData5 = _mm256_mask_load_epi32(storeData,mk8,q3+40);
        storeData6 = _mm256_mask_load_epi32(storeData,mk8,q3+48);
        storeData7 = _mm256_mask_load_epi32(storeData,mk8,q3+56);
        loadData = _mm256_mask_load_epi32(loadData,mk8,q2);
        loadData1 = _mm256_mask_load_epi32(loadData,mk8,q2+8);
        loadData2 = _mm256_mask_load_epi32(loadData,mk8,q2+16);
        loadData3 = _mm256_mask_load_epi32(loadData,mk8,q2+24);
        loadData4 = _mm256_mask_load_epi32(loadData,mk8,q2+32);
        loadData5 = _mm256_mask_load_epi32(loadData,mk8,q2+40);
        loadData6 = _mm256_mask_load_epi32(loadData,mk8,q2+48);
        loadData7 = _mm256_mask_load_epi32(loadData,mk8,q2+56);
        LoadData = _mm512_cvtepi32_epi64(loadData);
        LoadData1 = _mm512_cvtepi32_epi64(loadData1);
        LoadData2 = _mm512_cvtepi32_epi64(loadData2);
        LoadData3 = _mm512_cvtepi32_epi64(loadData3);
        LoadData4 = _mm512_cvtepi32_epi64(loadData4);
        LoadData5 = _mm512_cvtepi32_epi64(loadData5);
        LoadData6 = _mm512_cvtepi32_epi64(loadData6);
        LoadData7 = _mm512_cvtepi32_epi64(loadData7);

        vindex = _mm512_add_epi64(LoadData,myoffset);
        vindex1 = _mm512_add_epi64(LoadData1,myoffset);
        vindex2 = _mm512_add_epi64(LoadData2,myoffset);
        vindex3 = _mm512_add_epi64(LoadData3,myoffset);
        vindex4 = _mm512_add_epi64(LoadData4,myoffset);
        vindex5 = _mm512_add_epi64(LoadData5,myoffset);
        vindex6 = _mm512_add_epi64(LoadData6,myoffset);
        vindex7 = _mm512_add_epi64(LoadData7,myoffset);

        _mm512_i64scatter_epi32(local_pred,vindex,storeData,4);
        _mm512_i64scatter_epi32(local_pred,vindex1,storeData1,4);
        _mm512_i64scatter_epi32(local_pred,vindex2,storeData2,4);
        _mm512_i64scatter_epi32(local_pred,vindex3,storeData3,4);
        _mm512_i64scatter_epi32(local_pred,vindex4,storeData4,4);
        _mm512_i64scatter_epi32(local_pred,vindex5,storeData5,4);
        _mm512_i64scatter_epi32(local_pred,vindex6,storeData6,4);
        _mm512_i64scatter_epi32(local_pred,vindex7,storeData7,4);

        q3+=nBlockWidth;
        q2+=nBlockWidth;
    }

    for(int i=0;i<cntRem;i++)
    {
        local_pred[q2[i]+myrowoffset]=q3[i];
    }
}

void Pre_Compute()
{
    #ifdef DISPRED
    Init_Shared_process<int>(SHARE_FILE1,local_vert,1,localpredid,local_pred);
    MPI_Barrier(MPI_COMM_WORLD);
    IntV all_local_vert[num_procs];
    #ifdef _INT64_
    MPI_Allgather(&local_vert,1 ,MPI_LONG,all_local_vert,1,MPI_LONG,MPI_COMM_WORLD);
    #elif defined(_UNSIGNEDINT)
    MPI_Allgather(&local_vert,1 ,MPI_UNSIGNED,all_local_vert,1,MPI_UNSIGNED,MPI_COMM_WORLD);
    #else
    MPI_Allgather(&local_vert,1 ,MPI_INT,all_local_vert,1,MPI_INT,MPI_COMM_WORLD);
    #endif
    Attach_Shared_process<int>(SHARE_FILE1,all_local_vert,1,predid,pred);
    MPI_Barrier(MPI_COMM_WORLD);
    #else
    Init_Shared_total<int>(SHARE_FILE1,num_vertice,1,localpredid,local_pred,0);
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

    Init_Shared_process<packed_edge>(SHARE_FILE1,totalbuffer[ranks],1+num_procs,myrunbufferid,myrunbuffer);
    MPI_Barrier(MPI_COMM_WORLD);
    Attach_Shared_process<packed_edge>(SHARE_FILE1,totalbuffer,1+num_procs,runbufferid,runbuffer);
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
    for(IntV i=0;i<local_vert;i++) local_pred[i]=-1;
    #else
    for(int64_t i=myrowoffset;i<myrowend;i++) local_pred[i]=-1;
    #endif
    iteration=0;

    warm_up();
}

void warm_up()
{
    qc=0;
    if(Owner(root)==ranks)
    {
        #ifdef DISPRED
        int local=Local_V(root);
        #else
        int local=root;
        #endif
        q1[qc++]=local;
        local_pred[local]=root;
    }
    int64_t row_begin,row_end;
    int flag_b2u=1;
    nextactive=1;
    totalactive=1;
    while(flag_b2u>0)
    {
        #if defined(_PUSH_) && defined(_PULL_)
        if(nextactive < ALPHA * num_vertice)
        #endif
        {
            #ifdef _PUSH_
            memcpy(bufindex,bufferoffset,num_procs*sizeof(int64_t));
            int64_t tmp_bufindex[num_procs*num_procs];

            for(int i=0;i<qc;i++)
            {
                int localq = Local_V(q1[i]);
                row_begin = rowstarts[localq];
                row_end = rowstarts[localq+1];
                for(int64_t j=row_begin;j<row_end;j++)
                {
#ifdef DCOL_
                    int to_where = column[j].rk;
                    myrunbuffer[bufindex[to_where]].v0=Globalid(ranks,q1[i]);
                    myrunbuffer[bufindex[to_where]++].v1=Globalid(to_where,column[j].id);
#else
                    int to_where = Owner(column[j].id);
                    myrunbuffer[bufindex[to_where]].v0 = q1[i] + myrowoffset;
                    myrunbuffer[bufindex[to_where]++].v1 = column[j].id;
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
                    int local = Local_V(runbuffer[i][j].v1);
                    #else
                    int local = runbuffer[i][j].v1;
                    #endif
                    if(local_pred[local]<0) 
                    {
                        local_pred[local]=runbuffer[i][j].v0;
                        q1[qc++] = local;
                    }
                }
            }
            int64_t longqc = (int64_t)qc;
            MPI_Allreduce(&longqc,&nextactive,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
            totalactive+=nextactive;
            if(nextactive==0) flag_b2u=-1;
            #endif

        }
        #if defined(_PUSH_) && defined(_PULL_)
        else
        #endif
        {
            #ifdef _PULL_
                qc=0;
                #ifdef DISPRED
                for(IntV i=0;i<local_vert;i++)
                #else
                for(int64_t i=myrowoffset;i<myrowend;i++)
                #endif
                {
                    if(local_pred[i]==-1)
                    {
                        #ifdef DISPRED
                        q1[qc++]=i;
                        #else
                        q1[qc++]=i-myrowoffset;
                        #endif
                    }
                }

                while(1)
                {
                    int q3c=0;
                    for(int64_t i=0;i<qc;i++)
                    {
                        int localq = q1[i];
                        row_begin = outrowstarts[localq];
                        row_end = outrowstarts[localq+1];
                            for(int64_t j=row_begin;j<row_end;j++)
                            {
                                #ifdef DISPRED
                                int localc=outcolumn[j].id;
                                int ownerc=outcolumn[j].rk;
                                if(pred[ownerc][localc]>-1)
                                {
                                    /*pred[q1[i]]=column[j];*/
                                    q2[q3c]=q1[i];
                                    q3[q3c++] = Globalid(outcolumn[j]);
                                    break;
                                }
                                #else
#ifdef DCOL_
                                int global_temp=Globalid(outcolumn[j]);
#else
                                int global_temp = outcolumn[j].id;
#endif
                                if(local_pred[global_temp]>-1)
                                {
                                    /*pred[q1[i]]=column[j];*/
                                    q2[q3c]=q1[i];
                                    q3[q3c++] = global_temp;
                                    break;
                                }
                                #endif
                            }
                        
                    }
                    MPI_Barrier(MPI_COMM_WORLD);

                    for(int i=0;i<q3c;i++)
                    {
                        #ifdef DISPRED
                        local_pred[q2[i]]=q3[i];
                        #else
                        local_pred[q2[i]+myrowoffset]=q3[i];
                        #endif
                    }
                    
                    q2c=qc;
                    qc=0;
                    for(int i=0;i<q2c;i++)
                    {
                        if(local_pred[q1[i]+myrowoffset]==-1)
                        {
                            q2[qc++] = q1[i];
                        }
                    }

                    q2c = q2c -qc;

                    int* tmp =q1;q1=q2;q2=tmp;

                    int64_t longq2c=(int64_t)q2c;
                    MPI_Allreduce(&longq2c,&nextactive,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
                    totalactive+=nextactive;
                    if(nextactive==0) 
                    {
                        flag_b2u=-1;
                        break;
                    }
                }
            #endif
        }
    }

    #ifdef DISPRED
    for(IntV i=0;i<local_vert;i++)
    #else
    for(int64_t i=myrowoffset;i<myrowend;i++)
    #endif
    {
        if(local_pred[i]==-1) local_pred[i]--;
        else local_pred[i]=-1;
    }
}


void Compute()
{
    qc=0;
    if(Owner(root)==ranks)
    {
        #ifdef DISPRED
        int local=Local_V(root);
        #else
        int local=root;
        #endif
        q1[qc++]=local;
        local_pred[local]=root;
    }
    int64_t row_begin,row_end;
    int flag_b2u=1;
    nextactive=1;
    totalactive=1;
    if(ranks==0) printf("[T2D] Iteration : %d\n  Current_Active : %ld\n  Total_Active : %ld\n\n",iteration++,nextactive,totalactive);
    while(flag_b2u>0)
    {
        #if defined(_PUSH_) && defined(_PULL_)
        if(nextactive < ALPHA * num_vertice)
        #endif
        {
            #ifdef _PUSH_
            double b2utime=MPI_Wtime();
            memcpy(bufindex,bufferoffset,num_procs*sizeof(int64_t));
            int64_t tmp_bufindex[num_procs*num_procs] __attribute__((aligned(64)));

            for(int i=0;i<qc;i++)
            {
                int localq = q1[i]-myrowoffset;
                row_begin = rowstarts[localq];
                row_end = rowstarts[localq+1];
                for(int64_t j=row_begin;j<row_end;j++)
                {
#ifdef DCOL_
                    int to_where = column[j].rk;
                    myrunbuffer[bufindex[to_where]].v0=Globalid(ranks,q1[i]);
                    myrunbuffer[bufindex[to_where]++].v1=Globalid(to_where,column[j].id);
#else
                    int to_where = Owner(column[j].id);
                    myrunbuffer[bufindex[to_where]].v0 = q1[i];
                    myrunbuffer[bufindex[to_where]++].v1 = column[j].id;
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
                    int local = Local_V(runbuffer[i][j].v1);
                    #else
                    int local = runbuffer[i][j].v1;
                    #endif
                    if(local_pred[local]<0) 
                    {
                        local_pred[local]=runbuffer[i][j].v0;
                        q1[qc++] = local;
                    }
                }
            }
            int64_t longqc = (int64_t)qc;
            MPI_Allreduce(&longqc,&nextactive,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
            totalactive+=nextactive;
            if(ranks==0) printf("[T2D] Iteration : %d\n  Current_Active : %ld\n  Total_Active : %ld\n\n",iteration++,nextactive,totalactive);
            if(nextactive==0) flag_b2u=-1;
            b2utime = MPI_Wtime()-b2utime;
			if(ranks==0)	printf("[T2D] Iteration : %d Time : %f\n\n",iteration-1,b2utime);
            #endif

        }
        #if defined(_PUSH_) && defined(_PULL_)
        else
        #endif
        {
            #ifdef _PULL_
                qc=0;
                #ifdef DISPRED
                for(IntV i=0;i<local_vert;i++)
                #else
                for(int64_t i=myrowoffset;i<myrowend;i++)
                #endif
                {
                    if(local_pred[i]==-1)
                    {
                        q1[qc++]=i-myrowoffset;
                    }
                }

                while(1)
                {
                    double b2utime=MPI_Wtime();
                    int q3c=0;
                    for(int64_t i=0;i<qc;i++)
                    {
                        int localq = q1[i];
                        row_begin = outrowstarts[localq];
                        row_end = outrowstarts[localq+1];
                            for(int64_t j=row_begin;j<row_end;j++)
                            {
                                #ifdef DISPRED
                                int localc=outcolumn[j].id;
                                int ownerc=outcolumn[j].rk;
                                if(pred[ownerc][localc]>-1)
                                {
                                    q2[q3c]=localq;
                                    q3[q3c++] = Globalid(outcolumn[j]);
                                    break;
                                }
                                #else
#ifdef DCOL_
                                int global_temp = Globalid(outcolumn[j]);
#else
                                int global_temp = outcolumn[j].id;
#endif
                                if(local_pred[global_temp]>-1)
                                {
                                    q2[q3c]=q1[i];
                                    q3[q3c++] = global_temp;
                                    break;
                                }
                                #endif
                            }
                        
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                   /* double htime = MPI_Wtime();
                    myoffset = _mm512_set1_epi64(myrowoffset);
                    unsigned int mktemp = 0x0000ffff;
                    mk8 = _cvtu32_mask8(mktemp);
                    imm_vector(q2,q3,local_pred,q3c);
                    if(ranks==0) printf("write time: %f \n",MPI_Wtime()-htime);*/
                    for(int i=0;i<q3c;i++)
                    {
                        #ifdef DISPRED
                        local_pred[q2[i]]=q3[i];
                        #else
                        local_pred[q2[i] + myrowoffset]=q3[i];
                        #endif
                    }

                    q2c=qc;
                    qc=0;
                    for(int i=0;i<q2c;i++)
                    {
                        if(local_pred[q1[i]+myrowoffset]==-1)
                        {
                            q2[qc++] = q1[i];
                        }
                    }

                    q2c = q2c -qc;

                    int* tmp =q1;q1=q2;q2=tmp;

                    int64_t longq2c=(int64_t)q2c;
                    MPI_Allreduce(&longq2c,&nextactive,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
                    totalactive+=nextactive;
                    if(ranks==0) printf("[B2U] Iteration : %d\n  Current_Active : %ld\n  Total_Active : %ld\n\n",iteration++,nextactive,totalactive);
                    b2utime = MPI_Wtime()-b2utime;
				    if(ranks==0) printf("[B2U] Iteration : %d Time : %f\n\n",iteration-1,b2utime);
                    if(nextactive==0) 
                    {
                        flag_b2u=-1;
                        break;
                    }
                }
            #endif
        }
    }
}


void Free_Compute()
{
    #ifdef DISPRED
    Disattach_Shared_process<int>(localpredid,local_pred,predid,pred);
    #else
    Disattach_Shared_total<int>(localpredid,local_pred);
    #endif
#ifdef _PUSH_
    free(bufferoffset);
    free(bufindex);
    Disattach_Shared_process<packed_edge>(myrunbufferid,myrunbuffer,runbufferid,runbuffer);
    MPI_Barrier(MPI_COMM_WORLD);
    free(remoterunoffset);
#endif
	_mm_free(q1);
	_mm_free(q2);
    #ifdef _PULL_
    _mm_free(q3);
    #endif

    Free_Graph();
}

void Statistics_Print()
{
    return ;
}