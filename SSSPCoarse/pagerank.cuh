//
// Created by gxl on 2020/12/31.
//

#ifndef PTGRAPH_PAGERANK_CUH
#define PTGRAPH_PAGERANK_CUH
#include "common.cuh"

//#include "GraphMeta.cuh"
void conventionParticipatePR(string prPath);
void prShare();
void prShareByInDegree(string prPath);
void prOpt(string prPath, float adviseK);
void prOptSwap();
void prShareByInDegreeTrace(string prPath);
void conventionParticipatePRHalfStatic();
void prOptRandom(const string& prPath, float adviseK);
#endif //PTGRAPH_PAGERANK_CUH
