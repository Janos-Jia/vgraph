#pragma once
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <algorithm>
#include <stdint.h>
#include "IO.h"
#include "config.h"
#ifdef COUNTPCM
#include "utils.h"
#include "cpucounters.h"
#endif

void Pre_Compute();
void Compute();
void Free_Compute();
void Statistics_Print();

extern int ranks,num_procs;
extern int64_t num_vertice,totaledge,myrowoffset,myrowend;
extern IntV local_vert,numv_per_proc;

using namespace std;

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &ranks);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#ifdef COUNTPCM
    pcm::PCM* m;
    bool pcm_flag;
    pcm::SystemCounterState before_state, after_state;
    if (ranks == 0)
    {
        m = pcm::PCM::getInstance();
        if (m->program() != pcm::PCM::Success)
        {
            pcm_flag = false;
            cout << "Error in open PCM\n" << endl;
        }
        else pcm_flag = true;
    }
#endif
    num_vertice = atoll(argv[2]);
    double starttime=MPI_Wtime();
    Construct_Graph(argv[1]);
    if(ranks==0) cout<<"Finish Read"<<endl;
    Pre_Compute();
#ifdef COUNTPCM
    if (ranks == 0 && pcm_flag) before_state = pcm::getSystemCounterState();
#endif
    double pretime=MPI_Wtime()-starttime;
    Compute();
    double computetime=MPI_Wtime()-starttime-pretime;
#ifdef COUNTPCM
    if (ranks == 0 && pcm_flag) after_state = pcm::getSystemCounterState();
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    starttime=MPI_Wtime()-starttime;
    Statistics_Print();
    if(ranks==0) cout<<"Pre-Compute Time: "<<pretime<<endl<<"Compute Time: "<<computetime<<endl<<"  End-to-end Time: "<<starttime<<endl;
#ifdef COUNTPCM
    if (ranks == 0 && pcm_flag)
    {
        cout << "Instructions per clock: " << pcm::getIPC<pcm::SystemCounterState>(before_state, after_state) << endl;
        cout << "L3 Cache Hit Ratio: " << pcm::getL3CacheHitRatio<pcm::SystemCounterState>(before_state, after_state) << endl;
        cout << "QPI traffic: " << unit_format(getAllIncomingQPILinkBytes(before_state, after_state)) << endl;


        m->cleanup();

    }
#endif
    Free_Compute();
    MPI_Finalize();
    return 0;
}