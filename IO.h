#pragma once
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cstdlib>
#include <cmath>
#include <string.h>
#include <sys/mman.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <mpi.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "config.h"

using namespace std;

extern int ranks,num_procs;
extern int64_t num_vertice,totaledge,myrowoffset,myrowend;
extern IntV local_vert,numv_per_proc;

struct packed_edge
{
    int v0;
    int v1;
    #ifdef WEIGHTS_
    Vtype val;
    #endif
};

struct stat sb;
packed_edge * b_map;

#ifdef _PUSH_
int64_t * rowstarts;
int myrowid;

int * degree_all;
int * degree;
int degreeid;

col_type * column;
col_type * mycolumn;
int mycolid;

col_type ** remote_column;
int * remote_colshareid;
int64_t ** remote_rowstarts;
int * remote_rowshareid;

int64_t local_edge;



#endif

#if defined(_PULL_) && defined(DIRECTED_)
	int * outdegree_all;
	int * outdegree;
	int outdegreeid;
        
	int * outbufferindex;
	int  outbufferindexid;

	int64_t * outrowstarts;
	int outmyrowid;
	col_type * outcolumn;
	int outmycolid;
	col_type ** outremote_column;
	int * outremote_colshareid;
	int64_t ** outremote_rowstarts;
	int * outremote_rowshareid;

	int64_t outlocal_edge;



#endif

void init_row()
{
    #ifdef _PUSH_
	key_t key = ftok(SHARE_FILE0,ranks+num_procs);
	myrowid = shmget(key,(local_vert+1)*sizeof(int64_t),IPC_CREAT|IPC_EXCL|0666);
	if(myrowid==-1) printf("[Rank %d] init row shareid %d Error!!!\n",ranks,ranks);
	rowstarts = (int64_t*)shmat(myrowid,NULL,0);
    #endif
	#if defined(_PULL_) && defined(DIRECTED_)
	key_t key1 = ftok(SHARE_FILE0,ranks+num_procs*3);
	outmyrowid = shmget(key1,(local_vert+1)*sizeof(int64_t),IPC_CREAT|IPC_EXCL|0666);
	if(outmyrowid==-1) printf("[Rank %d] init row outshareid %d Error!!!\n",ranks,ranks);
	outrowstarts = (int64_t*)shmat(outmyrowid,NULL,0);
	#endif


	MPI_Barrier(MPI_COMM_WORLD);
}

void attach_row()
{
    #ifdef _PUSH_
	remote_rowshareid=(int*)malloc(num_procs*sizeof(int));
	remote_rowstarts = (int64_t**)malloc(num_procs*sizeof(int64_t*));

	for(int i=0;i<num_procs;i++)
	{
		key_t key = ftok(SHARE_FILE0,i+num_procs);
		if(i==(num_procs-1)) 
		remote_rowshareid[i]=shmget(key,(num_vertice - (num_procs-1) * numv_per_proc +1)*sizeof(int64_t),IPC_CREAT|0666);
		else
		remote_rowshareid[i]=shmget(key,(local_vert+1)*sizeof(int64_t),IPC_CREAT|0666);
		if(remote_rowshareid[i]==-1) printf("[Rank %d] attach remote row shareid %d Error!!!\n",ranks,ranks);
		remote_rowstarts[i]=(int64_t*)shmat(remote_rowshareid[i],NULL,0); 
	}
    #endif
	#if defined(_PULL_) && defined(DIRECTED_)
	outremote_rowshareid=(int*)malloc(num_procs*sizeof(int));
	outremote_rowstarts = (int64_t**)malloc(num_procs*sizeof(int64_t*));

	for(int i=0;i<num_procs;i++)
	{
		key_t key1 = ftok(SHARE_FILE0,i+num_procs*3);
		if(i==(num_procs-1)) 
		outremote_rowshareid[i]=shmget(key1,(num_vertice - (num_procs-1) * numv_per_proc +1)*sizeof(int64_t),IPC_CREAT|0666);
		else
		outremote_rowshareid[i]=shmget(key1,(local_vert+1)*sizeof(int64_t),IPC_CREAT|0666);
		if(outremote_rowshareid[i]==-1) printf("[Rank %d] attach remote outrow shareid %d Error!!!\n",ranks,ranks);
		outremote_rowstarts[i]=(int64_t*)shmat(outremote_rowshareid[i],NULL,0); 
	}
	#endif
	MPI_Barrier(MPI_COMM_WORLD);
}

void dis_row()
{
    #ifdef _PUSH_
	shmctl(myrowid,IPC_RMID,0);
	for(int i=0;i<num_procs;i++)
	{
		shmctl(remote_rowshareid[i],IPC_RMID,0);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	shmdt((const void *)rowstarts);
	for(int i=0;i<num_procs;i++)
	{
		shmdt((const void *)remote_rowstarts[i]);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	shmctl(myrowid,IPC_RMID,0);
	for(int i=0;i<num_procs;i++)
	{
		shmctl(remote_rowshareid[i],IPC_RMID,0);
	}
    #endif
	#if defined(_PULL_) && defined(DIRECTED_)
	shmctl(outmyrowid,IPC_RMID,0);
	for(int i=0;i<num_procs;i++)
	{
		shmctl(outremote_rowshareid[i],IPC_RMID,0);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	shmdt((const void *)outrowstarts);
	for(int i=0;i<num_procs;i++)
	{
		shmdt((const void *)outremote_rowstarts[i]);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	shmctl(outmyrowid,IPC_RMID,0);
	for(int i=0;i<num_procs;i++)
	{
		shmctl(outremote_rowshareid[i],IPC_RMID,0);
	}
	#endif
}

void init_col()
{
    #ifdef _PUSH_
	key_t key = ftok(SHARE_FILE0,ranks+num_procs*2);
	mycolid = shmget(key,(local_edge)*sizeof(col_type),IPC_CREAT|IPC_EXCL|0666);
	if(mycolid==-1) printf("[Rank %d] init col shareid %d Error!!!\n",ranks,ranks);
	column = (col_type *)shmat(mycolid,NULL,0);
    #endif
	#if defined(_PULL_) && defined(DIRECTED_)
	key_t key1 = ftok(SHARE_FILE0,ranks+num_procs*4);
	outmycolid = shmget(key1,(outlocal_edge)*sizeof(col_type),IPC_CREAT|IPC_EXCL|0666);
	if(outmycolid==-1) printf("[Rank %d] init outcol shareid %d Error!!!\n",ranks,ranks);
	outcolumn = (col_type *)shmat(outmycolid,NULL,0);
	#endif
	MPI_Barrier(MPI_COMM_WORLD);
}

void attach_col()
{
    #ifdef _PUSH_
	remote_colshareid=(int*)malloc(num_procs*sizeof(int));
	remote_column = (col_type**)malloc(num_procs*sizeof(col_type*));

	int64_t all_localedge[num_procs];
	MPI_Allgather(&local_edge,1,MPI_LONG,&all_localedge,1,MPI_LONG,MPI_COMM_WORLD);

	for(int i=0;i<num_procs;i++)
	{
		key_t key = ftok(SHARE_FILE0,i+num_procs*2);
		remote_colshareid[i]=shmget(key,(all_localedge[i])*sizeof(col_type),IPC_CREAT|0666);
		if(remote_colshareid[i]==-1) printf("[Rank %d] attach remote col shareid %d Error!!!\n",ranks,ranks);
		remote_column[i]=(col_type *)shmat(remote_colshareid[i],NULL,0); 
	}
    #endif
	#if defined(_PULL_) && defined(DIRECTED_)
	outremote_colshareid=(int*)malloc(num_procs*sizeof(int));
	outremote_column = (col_type**)malloc(num_procs*sizeof(col_type*));

	int64_t outall_localedge[num_procs];
	MPI_Allgather(&outlocal_edge,1,MPI_LONG,&outall_localedge,1,MPI_LONG,MPI_COMM_WORLD);

	for(int i=0;i<num_procs;i++)
	{
		key_t key1 = ftok(SHARE_FILE0,i+num_procs*4);
		outremote_colshareid[i]=shmget(key1,(outall_localedge[i])*sizeof(col_type),IPC_CREAT|0666);
		if(outremote_colshareid[i]==-1) printf("[Rank %d] attach remote outoutcol shareid %d Error!!!\n",ranks,ranks);
		outremote_column[i]=(col_type *)shmat(outremote_colshareid[i],NULL,0); 
	}
	#endif
	MPI_Barrier(MPI_COMM_WORLD);
	
}

void dis_col()
{
    #ifdef _PUSH_
	shmctl(mycolid,IPC_RMID,0);
	for(int i=0;i<num_procs;i++)
	{
		shmctl(remote_colshareid[i],IPC_RMID,0);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	shmdt((const void *)column);
	for(int i=0;i<num_procs;i++)
	{
		shmdt((const void *)remote_column[i]);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	shmctl(mycolid,IPC_RMID,0);
	for(int i=0;i<num_procs;i++)
	{
		shmctl(remote_colshareid[i],IPC_RMID,0);
	}
    #endif
	#if defined(_PULL_) && defined(DIRECTED_)
		shmctl(outmycolid,IPC_RMID,0);
	for(int i=0;i<num_procs;i++)
	{
		shmctl(outremote_colshareid[i],IPC_RMID,0);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	shmdt((const void *)outcolumn);
	for(int i=0;i<num_procs;i++)
	{
		shmdt((const void *)outremote_column[i]);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	shmctl(outmycolid,IPC_RMID,0);
	for(int i=0;i<num_procs;i++)
	{
		shmctl(outremote_colshareid[i],IPC_RMID,0);
	}
	#endif
}




void Construct_Graph(char * filename)
{
    #ifdef _PUSH_
    if(ranks==0)
	{
		key_t key = ftok(SHARE_FILE0,-1+num_procs);
		degreeid=shmget(key,(num_vertice)*sizeof(int),IPC_CREAT|IPC_EXCL|0666);
		if(degreeid==-1) printf("[Rank %d] creat degree shareid %d Error!!!\n",ranks,ranks);
		degree_all = (int *)shmat(degreeid,NULL,0); 
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if(ranks!=0)
	{
		key_t key = ftok(SHARE_FILE0,-1+num_procs);
		degreeid=shmget(key,(num_vertice)*sizeof(int),IPC_CREAT|0666);
		if(degreeid==-1) printf("[Rank %d] creat degree shareid %d Error!!!\n",ranks,ranks);
		degree_all = (int *)shmat(degreeid,NULL,0); 
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if(ranks==0) memset(degree_all,0,num_vertice*sizeof(int));
    #endif
    #if defined(_PULL_) && defined(DIRECTED_)
    int halfranks=num_procs>>1;
	if(ranks==halfranks)
	{
		key_t key = ftok(SHARE_FILE0,-2+num_procs);
		outdegreeid=shmget(key,(num_vertice)*sizeof(int),IPC_CREAT|IPC_EXCL|0666);
		if(outdegreeid==-1) printf("[Rank %d] creat degree shareid %d Error!!!\n",ranks,ranks);
		outdegree_all = (int *)shmat(outdegreeid,NULL,0); 
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if(ranks!=halfranks)
	{
		key_t key = ftok(SHARE_FILE0,-2+num_procs);
		outdegreeid=shmget(key,(num_vertice)*sizeof(int),IPC_CREAT|0666);
		if(outdegreeid==-1) printf("[Rank %d] creat degree shareid %d Error!!!\n",ranks,ranks);
		outdegree_all = (int *)shmat(outdegreeid,NULL,0); 
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if(ranks==halfranks) memset(outdegree_all,0,num_vertice*sizeof(int));
    #endif
    MPI_Barrier(MPI_COMM_WORLD);

    /*Open File*/
	int fd;
	fd = open(filename, O_RDONLY);
	if (fd == -1) {
		printf("open Error\n");
		exit(-1);
	}
    if (fstat(fd, &sb) == -1) {
		printf("fstat Error\n");
		exit(-1);   
	}
    b_map = (packed_edge*)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	totaledge = sb.st_size/sizeof(packed_edge);
	if(ranks==0) printf("Total Edge : %ld \n",totaledge);
	close(fd);
    int64_t myfileoff = (totaledge/num_procs)*ranks;
	int64_t myfileend;
	if(ranks!=(num_procs-1))
	{
		myfileend = (totaledge/num_procs)*(ranks+1);
	}
	else
	{
		myfileend = totaledge;
	}

    double Construct_Time=MPI_Wtime();
#ifdef OMP_
#pragma omp parallel for
#endif
	for(int64_t i=myfileoff;i<myfileend;i++)
	{
		if(b_map[i].v0!=b_map[i].v1)
        {
            #ifdef _PUSH_
			__sync_fetch_and_add(&degree_all[b_map[i].v0],1);
#ifdef D_CC
			__sync_fetch_and_add(&degree_all[b_map[i].v1], 1);
#endif
            #endif
			#if defined(_PULL_) && defined(DIRECTED_)
			__sync_fetch_and_add(&outdegree_all[b_map[i].v1],1);
			#endif
		}
	}
    Construct_Time=MPI_Wtime()-Construct_Time;
    if(ranks==0) printf("First Read Time: %f\n",Construct_Time);

    /*Prepare for Second Read*/
    local_vert=num_vertice/num_procs;
	if(num_vertice%num_procs) local_vert++;
	numv_per_proc = local_vert;
	if(ranks==num_procs-1) local_vert = num_vertice - (num_procs-1) * numv_per_proc ;
	myrowoffset = ranks * (numv_per_proc);
	myrowend = myrowoffset+local_vert;

    init_row();
	attach_row();

    #ifdef _PUSH_
    rowstarts[0]=0;
	for(IntV i=0;i<local_vert;i++)
	{
		rowstarts[i+1] = rowstarts[i]+degree_all[myrowoffset+i];
	}
    local_edge = rowstarts[local_vert];
    #endif

    #if defined(_PULL_) && defined(DIRECTED_)
    outrowstarts[0]=0;
	for(IntV i=0;i<local_vert;i++)
	{
		outrowstarts[i+1] = outrowstarts[i]+outdegree_all[myrowoffset+i];
	}
    outlocal_edge = outrowstarts[local_vert];
    #endif
	init_col();
	attach_col(); 
    
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef _PUSH_
    if(ranks==0) memset(degree_all,0,num_vertice*sizeof(int));
    #endif
    #if defined(_PULL_) && defined(DIRECTED_)
    if(ranks == (num_procs>>1)) memset(outdegree_all,0,num_vertice*sizeof(int));
    #endif
    MPI_Barrier(MPI_COMM_WORLD);
    /*Begin Second Read*/
    Construct_Time=MPI_Wtime();
#ifdef OMP_
#pragma omp parallel for
#endif
    for(int64_t i=myfileoff;i<myfileend;i++)
	{
		if(b_map[i].v0!=b_map[i].v1)
        {
            #ifdef _PUSH_
			int onw = Owner(b_map[i].v0);
			int localid=b_map[i].v0-onw*numv_per_proc;
            int tmp = __sync_fetch_and_add(&degree_all[b_map[i].v0],1);
			int64_t tmpt = remote_rowstarts[onw][localid]+tmp;
			#ifdef WEIGHTS_
			write_col(remote_column[onw][tmpt], b_map[i].v1, b_map[i].val);
			#else	
			write_col(remote_column[onw][tmpt], b_map[i].v1);
			#endif
#ifdef D_CC
			int onw1 = Owner(b_map[i].v1);
			int localid1 = b_map[i].v1 - onw1 * numv_per_proc;
			int tmp1 = __sync_fetch_and_add(&degree_all[b_map[i].v1], 1);
			int64_t tmpt1 = remote_rowstarts[onw1][localid1] + tmp1;
#ifdef WEIGHTS_
			write_col(remote_column[onw1][tmpt1], b_map[i].v0, b_map[i].val);
#else	
			write_col(remote_column[onw1][tmpt1], b_map[i].v0);
#endif
#endif
            #endif
			#if defined(_PULL_) && defined(DIRECTED_)
			int outonw = Owner(b_map[i].v1);
			int outlocalid=b_map[i].v1-outonw*numv_per_proc;
            int outtmp = __sync_fetch_and_add(&outdegree_all[b_map[i].v1],1);
			int64_t outtmpt = outremote_rowstarts[outonw][outlocalid]+outtmp;
            
			#ifdef WEIGHTS_
			write_col(outremote_column[outonw][outtmpt], b_map[i].v0, b_map[i].val);
			#else
			write_col(outremote_column[outonw][outtmpt], b_map[i].v0);
			#endif
			#endif
        }
	}
	
	Construct_Time=MPI_Wtime()-Construct_Time;

	if(ranks==0) printf("Second Read Time: %f\n",Construct_Time);
	/* TEST CODE
	if (ranks == 0)
	{
		for (int i = rowstarts[17]; i < rowstarts[18]; i++)
		{
			printf("%d ", column[i]);
		}
		printf("\n");

		for (int i = outrowstarts[17]; i < outrowstarts[18]; i++)
		{
			printf("%d ", outcolumn[i]);
		}
		printf("\n");
	}
	*/
    #ifdef _PUSH_
    shmdt((const void *)degree_all);
	MPI_Barrier(MPI_COMM_WORLD);
		
	shmctl(degreeid,IPC_RMID,0);
    #endif
	#if defined(_PULL_) && defined(DIRECTED_)
	shmdt((const void *)outdegree_all);
	MPI_Barrier(MPI_COMM_WORLD);
		
	shmctl(outdegreeid,IPC_RMID,0);
	#endif

	munmap(b_map,sb.st_size);
}


void Free_Graph()
{
    dis_row();
    dis_col();

    #ifdef _PUSH_
    free(remote_colshareid);
	free(remote_column);
	free(remote_rowstarts);
	free(remote_rowshareid);
    #endif
    #if defined(_PULL_) && defined(DIRECTED_)
    free(outremote_colshareid);
	free(outremote_column);
	free(outremote_rowstarts);
	free(outremote_rowshareid);
    #endif
}

template <class E>
void Init_Shared_process(char * filename, int64_t length, int offset, int &shareid, E * &sharearray)
{
	key_t key = ftok(filename,offset+ranks);
	shareid = shmget(key,(length)*sizeof(E),IPC_CREAT|IPC_EXCL|0666);
	if(shareid==-1) cout<<"[Rank "<<ranks<<"] creat sharearray Error"<<endl;
	sharearray = (E*)shmat(shareid,NULL,0); 
}

template <class E>
void Attach_Shared_process(char * filename, int64_t * lengtharray, int offset, int * &shareidarray, E ** &remote_sharearray)
{
	shareidarray=(int *)malloc(num_procs*sizeof(int));
	remote_sharearray=(E **)malloc(num_procs*sizeof(E*));
	for(int i=0;i<num_procs;i++)
	{
		key_t key = ftok(filename,i+offset);
		shareidarray[i] = shmget(key,(lengtharray[i])*sizeof(E),IPC_CREAT|0666);
		if(shareidarray[i]==-1) cout<<"[Rank "<<ranks<<"] attach sharearray Error"<<endl;
		remote_sharearray[i]=(E*)shmat(shareidarray[i],NULL,0);
	}
}

template <class E>
void Disattach_Shared_process(int &shareid, E * &sharearray, int * &shareidarray, E ** &remote_sharearray)
{
	shmdt((const void *)sharearray);
	for(int i=0;i<num_procs;i++)
	{
		shmdt((const void *)(remote_sharearray[i]));
	}
	MPI_Barrier(MPI_COMM_WORLD);

	shmctl(shareid,IPC_RMID,0);
	for(int i=0;i<num_procs;i++)
	{
		shmctl(shareidarray[i],IPC_RMID,0);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

template <class E>
void Init_Shared_total(char * filename, int64_t length, int offset, int &shareid, E * &sharearray, int mainrank)
{
	key_t key = ftok(filename,offset);
	if(ranks == mainrank)
	{
		shareid = shmget(key,(length)*sizeof(E),IPC_CREAT|IPC_EXCL|0666);
		if(shareid==-1) cout<<"[Rank "<<ranks<<"] creat sharearray Error"<<endl;
		sharearray = (E*)shmat(shareid,NULL,0); 
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if(ranks != mainrank)
	{
		shareid = shmget(key,(length)*sizeof(E),IPC_CREAT|0666);
		if(shareid==-1) cout<<"[Rank "<<ranks<<"] attach sharearray Error"<<endl;
		sharearray = (E*)shmat(shareid,NULL,0); 
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

template <class E>
void Disattach_Shared_total(int &shareid, E * &sharearray)
{
	shmdt((const void *)sharearray);
	MPI_Barrier(MPI_COMM_WORLD);
	shmctl(shareid,IPC_RMID,0);
	MPI_Barrier(MPI_COMM_WORLD);
}


template <class T>
inline bool cas(T * ptr, T old_val, T new_val) {
    return __sync_bool_compare_and_swap((int*)ptr, *((int*)&old_val), *((int*)&new_val));
}

template <class T>
inline bool cas8(T * ptr, T old_val, T new_val) {
    return __sync_bool_compare_and_swap((long*)ptr, *((long*)&old_val), *((long*)&new_val));
}
