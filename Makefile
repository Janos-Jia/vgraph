CFLAGS = -fopenmp -Wall -Drestrict=__restrict__ -DNDEBUG -mavx512f -mavx512vl -mavx512dq -ffast-math -fopenmp -DGRAPH_GENERATOR_MPI   # -g -pg
LDFLAGS = -w -D_PULL_
#MPICC = /home/gs/jmh/openmp/bin/mpicxx -g -O3
MPICC = mpicxx -g -O3
APPS = BFS BFS-pull BFS-push D-BFS CC D-CC PR D-PR OD-PR O-PR SSSP SSSP-push CF O-CF
all: clean $(APPS)

SOURCES =
HEADERS = *.h

ifdef QPIVGRAPH
PCMPATH = 
CFLAGS += -DCOUNTPCM -I $(PCMPATH) -lpcm
endif


BFS: $(SOURCES) $(HEADERS)
	                $(MPICC) $(CFLAGS) $(LDFLAGS) $(SOURCES) $(GENERATOR_SOURCES) -D_PUSH_ BFS.cpp -lm -o BFS

BFS-push: $(SOURCES) $(HEADERS)
	                $(MPICC) $(CFLAGS) -w $(SOURCES) $(GENERATOR_SOURCES) -D_PUSH_ BFS.cpp -lm -o BFS-push

BFS-pull: $(SOURCES) $(HEADERS)
	                $(MPICC) $(CFLAGS) -w $(SOURCES) $(GENERATOR_SOURCES) -D_PULL_ -DDIRECTED_ BFS.cpp -lm -o BFS-pull

D-BFS: $(SOURCES) $(HEADERS)
	                $(MPICC) $(CFLAGS) $(LDFLAGS) $(SOURCES) $(GENERATOR_SOURCES) -D_PUSH_ -DDIRECTED_ BFS.cpp -lm -o D-BFS

CC: $(SOURCES) $(HEADERS)
	                $(MPICC) $(CFLAGS) $(LDFLAGS) $(SOURCES) $(GENERATOR_SOURCES) -D_PUSH_ CC.cpp -lm -o CC

D-CC: $(SOURCES) $(HEADERS)
	                $(MPICC) $(CFLAGS) $(LDFLAGS) $(SOURCES) $(GENERATOR_SOURCES) -D_PUSH_ -DD_CC CC.cpp -lm -o D-CC

PR: $(SOURCES) $(HEADERS)
	                $(MPICC) $(CFLAGS) $(LDFLAGS) $(SOURCES) $(GENERATOR_SOURCES) -D_PUSH_   PageRank.cpp -lm -o PR

D-PR: $(SOURCES) $(HEADERS)
	                $(MPICC) $(CFLAGS) $(LDFLAGS) $(SOURCES) $(GENERATOR_SOURCES) -D_PUSH_ -DDIRECTED_  PageRank.cpp -lm -o D-PR

O-PR: $(SOURCES) $(HEADERS)
	                $(MPICC) $(CFLAGS) $(LDFLAGS) $(SOURCES) $(GENERATOR_SOURCES) -D_PUSH_  -DOMP_  PageRank.cpp -lm -o O-PR

OD-PR: $(SOURCES) $(HEADERS)
	                $(MPICC) $(CFLAGS) $(LDFLAGS) $(SOURCES) $(GENERATOR_SOURCES) -D_PUSH_ -DOMP_ -DDIRECTED_  PageRank.cpp -lm -o OD-PR


SSSP: $(SOURCES) $(HEADERS)
	                $(MPICC) $(CFLAGS) $(LDFLAGS) $(SOURCES) $(GENERATOR_SOURCES) -D_PUSH_ -DWEIGHTS_ SSSP.cpp -lm -o SSSP

SSSP-push: $(SOURCES) $(HEADERS)
	                $(MPICC) $(CFLAGS) -w $(SOURCES) $(GENERATOR_SOURCES) -D_PUSH_ -DWEIGHTS_ SSSP.cpp -lm -o SSSP-push

CF: $(SOURCES) $(HEADERS)
	                $(MPICC) $(CFLAGS) $(LDFLAGS) $(SOURCES) $(GENERATOR_SOURCES) -D_PUSH_ -DWEIGHTS_ -D__INTVALUE_ CF.cpp -lm -o CF

O-CF: $(SOURCES) $(HEADERS)
	                $(MPICC) $(CFLAGS) $(LDFLAGS) $(SOURCES) $(GENERATOR_SOURCES) -D_PUSH_ -DWEIGHTS_ -D__INTVALUE_ -DOMP_ CF.cpp -lm -o O-CF

clean:
	                -rm -f $(APPS) *.o *.
