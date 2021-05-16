#include <sstream>
#include "pagerank.cuh"

int main(int argc, char** argv) {
    cudaFree(0);
    ArgumentParser arguments(argc, argv, true);
    if (arguments.input.empty()) {
        arguments.input = "/home/gxl/forTest/LiveJournal/raw/liveJournal.bcsc";
    }
    if (arguments.sourceNode == 0) {
        arguments.sourceNode = 0;
    }
    arguments.method = 0;
    if (arguments.method == 0) {
        conventionParticipatePR(arguments.input);
    } else if (arguments.method == 1){
        prShareByInDegreeTrace(arguments.input);
    } else if (arguments.method == 2){
        prOpt(arguments.input, arguments.adviseK);
    }
    return 0;
}