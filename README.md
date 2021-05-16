# fortest

GridGraph:
Open the GridGraphBFS
BFS-
./bin/bfs [path] [start vertex id] [memory budget]
Open the GridGraphPR
PageRank-
./bin/pagerank [path] [number of iterations] [memory budget]

Open the BFSCoarse
CoarseBFS:
./BFSCoarse --input [path] --source source

Open the PRCoarse
CoarsePageRank
./PRCoarse --input [path]

Open the SSSPCoarse
CoarseSSSP:
./SSSPCoarse --input [path] --source source

Open the DTNN
./DTNN


Open the ptGraph
BFS_opt
./ptGraph --type [bfs;sssp;pr] --source source --input [path]

SSSP_opt
./ptGraph --type [bfs;sssp;pr] --source source --input [path]

PR_opt
./ptGraph --type [bfs;sssp;pr] --input [path]
