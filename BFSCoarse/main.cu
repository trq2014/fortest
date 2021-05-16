#include <sstream>
#include "bfs.cuh"

int main(int argc, char** argv) {
    cudaFree(0);
    ArgumentParser arguments(argc, argv, true);
    if (arguments.input.empty()) {
        arguments.input = "/home/gxl/forTest/LiveJournal/raw/liveJournal.bcsr";
    }
    if (arguments.sourceNode == 0) {
        arguments.sourceNode = 0;
    }
    conventionParticipateBFS(arguments.input, arguments.sourceNode);
    return 0;
}