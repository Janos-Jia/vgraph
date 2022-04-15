Compilation:
----------------------------------------------------
*MPI >= 3.4.2 (with OpenMP supprot)
*g++ >= 7.5.0

----------------------------------------------------
to build:

1. Record the path to file "share_memory" and "share_memory_pred", and revise Environment Variable in "config.h"
	SHARE_FILE0 = $/path/to/$share_memory
	SHARE_FILE1 = $/path/to/$share_memory_pred

2. Then execute
	
	make
for compiling all APPs
or compling a certain APP by
	make BFS

----------------------------------------------------
To Run:

mpiexec -n x ./[APP] [path] [vertices]

*[APP]* appoints specific applications, include BFS, CC, PageRank, SSSP and CF.
*[path]* gives the path of an input graph, i.e. a file stored on a *shared* file system, consisting of *|E|* \<source vertex id, destination vertex id, edge data\> tuples in binary.
*[vertices]* gives the number of vertices *|V|*. Vertex IDs are represented with 32-bit integers and edge data can be omitted for unweighted graphs (e.g. the above applications except SSSP).
Note: CC makes the input graph undirected by adding a reversed edge to the graph for each loaded one; SSSP uses *float* as the type of weights.



