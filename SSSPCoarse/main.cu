#include <sstream>
#include "pagerank.cuh"

#include "sssp.cuh"
int main(int argc, char** argv) {
    cudaFree(0);
    ArgumentParser arguments(argc, argv, true);
    if (arguments.input.empty()) {
        arguments.input = ssspGraphPath;
    }
    if (arguments.sourceNode == 0) {
        arguments.sourceNode = 25838548;
    }
    arguments.method = 2;
    if (arguments.method == 0) {
        conventionParticipateSSSP(arguments.sourceNode, arguments.input);
    } else if (arguments.method == 1){
        ssspShareTrace(arguments.sourceNode, arguments.input);
    } else if (arguments.method == 2){
        //arguments.adviseK = 0.8;
        ssspOpt(arguments.sourceNode, arguments.input, arguments.adviseK);
    }
    return 0;
}