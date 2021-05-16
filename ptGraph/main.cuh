/*
#include <iostream>
#include <chrono>
#include <fstream>
#include <math.h>
#include "gpu_kernels.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <algorithm>
#include <thread>

using namespace std;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void convertBncr(uint vertexNum, ulong edgeNum, uint *nodePointer, uint *edgeList);

void
readGraphFromJava(string filePath, uint &testNumNodes, ulong &testNumEdge, ulong *nodePointersUL, uint *nodePointersI,
                  uint *edgeList, bool isNeedConvert);

void caculateInCommon(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);

long bfsCaculateInShare(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);
long
bfsCaculateInShareReturnValue(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode,
                              uint** bfsValue, int index);
long bfsCaculateInOpt(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);

long bfsCaculateInAsync(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);

long bfsCaculateInAsyncSwap(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);

long
bfsCaculateInAsyncSwapOpt(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);

long
bfsCaculateInAsyncOverloadWithSwap(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);

long
bfsCaculateInAsyncSwapOptWithOverload(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList,
                                      uint sourceNode);

long testCorruption(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);

long
bfsCaculateInAsyncSwapManage(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);

long ssspCaculateInShare(uint testNumNodes, uint testNumEdge, uint *nodePointersI, EdgeWithWeight *edgeList,
                         uint sourceNode);

long
ssspCaculateInOpt(uint testNumNodes, uint testNumEdge, uint *nodePointersI, EdgeWithWeight *edgeList, uint sourceNode);
long ssspCaculateCommonMemoryInnerAsync(uint testNumNodes, uint testNumEdge, uint *nodePointersI, EdgeWithWeight *edgeList,
                                   uint sourceNode);

long ccCaculateInShare(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);

long ccCaculateInOpt(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);

long prCaculateInShare(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);

long prCaculateInOpt(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);

void caculateInShareOpt(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);

void caculateInOptChooseByDegree(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);

long ccCaculateCommonMemoryInnerAsync(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);
long bfsCaculateInAsyncNoUVM(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);
long
bfsCaculateInAsyncNoUVMSwap(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);
void testBFS();

void testSSSP();

void testCC();

void testPagerank();
void conventionParticipateBFS();

void getMaxPartitionSize(unsigned long &max_partition_size, uint testNumNodes);
void getMaxPartitionSize(unsigned long &max_partition_size, unsigned long &totalSize, uint testNumNodes, float param, int edgeSize, int nodeParamSize = 15);
void getMaxPartitionSizeWithEdgeWeight(unsigned long &max_partition_size, uint testNumNodes);
long ccCaculateAsyncSwap(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);

struct PartEdgeListInfo {
    uint partActiveNodeNums;
    uint partEdgeNums;
    uint partStartIndex;
};

struct CommonPartitionInfo {
    uint startVertex;
    uint endVertex;
    uint nodePointerOffset;
    uint partitionEdgeSize;
};

void checkNeedTransferPartition(bool* needTransferPartition, CommonPartitionInfo* partitionInfoList, bool* isActiveNodeList, int partitionNum, uint testNumNodes, uint & activeNum) {
    uint tempMinNode = UINT_MAX;
    uint tempMaxNode = 0;
    for (uint j = 0; j < testNumNodes; j++) {
        if (isActiveNodeList[j]) {
            if (j < tempMinNode) {
                tempMinNode = j;
            }
            if (j > tempMaxNode) {
                tempMaxNode = j;
            }
            activeNum++;
        }
    }
    if (activeNum <= 0) {
        return;
    }
    for (int i = 0; i < partitionNum; i++) {
        needTransferPartition[i] = false;
        if (partitionInfoList[i].startVertex <= tempMaxNode && partitionInfoList[i].endVertex >= tempMinNode) {
            needTransferPartition[i] = true;
        }
    }
}

void caculatePartInfoForEdgeList(uint* overloadNodePointers,uint* overloadNodeList, uint* degree, vector<PartEdgeListInfo> & partEdgeListInfoArr, uint overloadNodeNum, uint overloadMemorySize, uint overloadSize) {
    partEdgeListInfoArr.clear();
    if (overloadMemorySize < overloadSize) {
        uint left = 0;
        uint right = overloadNodeNum - 1;
        while ((overloadNodePointers[right] + degree[overloadNodeList[right]] - overloadNodePointers[left]) > overloadMemorySize) {
            uint start = left;
            uint end = right;
            uint mid;
            while (start <= end) {
                mid = (start + end) / 2;
                uint headDistance = overloadNodePointers[mid] - overloadNodePointers[left];
                uint tailDistance = overloadNodePointers[mid] + degree[overloadNodeList[mid]] - overloadNodePointers[left];
                if (headDistance <= overloadMemorySize && tailDistance > overloadMemorySize) {
                    break;
                } else if (tailDistance <= overloadMemorySize) {
                    start = mid + 1;
                } else if (headDistance > overloadMemorySize) {
                    end = mid - 1;
                }
            }
            PartEdgeListInfo info;
            info.partActiveNodeNums = mid - left;
            info.partEdgeNums = overloadNodePointers[mid] - overloadNodePointers[left];
            info.partStartIndex = left;
            partEdgeListInfoArr.push_back(info);
            left = mid;
        }
        PartEdgeListInfo info;
        info.partActiveNodeNums = right - left + 1;
        info.partEdgeNums = overloadNodePointers[right] + degree[overloadNodeList[right]] - overloadNodePointers[left];
        info.partStartIndex = left;
        partEdgeListInfoArr.push_back(info);
    } else {
        PartEdgeListInfo info;
        info.partActiveNodeNums = overloadNodeNum;
        info.partEdgeNums = overloadSize;
        info.partStartIndex = 0;
        partEdgeListInfoArr.push_back(info);
    }
}

enum ALGORITHM {BFS, SSSP, CC};

uint fragment_size = 4096;
//string testGraphPath = "/home/gxl/labproject/subway/uk-2007-04/output.txt";
string converPath = "/home/gxl/labproject/subway/uk-2007-04/uk-2007-04.bcsr";
//string testGraphPath = "/home/gxl/labproject/subway/uk-2007-04/uk-2007-04.bcsr";
//string testGraphPath = "/home/gxl/labproject/subway/uk-2007Restruct.bcsr";
string testGraphPath = "/home/gxl/labproject/subway/friendster.bcsr";
//string testGraphPath = "/home/gxl/labproject/subway/friendsterRestruct.bcsr";
string testWeightGraphPath = "/home/gxl/labproject/subway/sk-2005.bwcsr";
string randomDataPath = "/home/gxl/labproject/subway/friendsterChange.random";

uint DIST_INFINITY = std::numeric_limits<unsigned int>::max() - 1;

void convertBwcsr();





class ArgumentParser {
private:

public:
    int argc;
    char **argv;

    bool canHaveSource;

    bool hasInput;
    bool hasSourceNode;
    string input;
    float adviseK;
    int sourceNode;
    int method;

    ArgumentParser(int argc, char **argv, bool canHaveSource);

    bool Parse();

    string GenerateHelpString();

};



ArgumentParser::ArgumentParser(int argc, char **argv, bool canHaveSource) {
    this->argc = argc;
    this->argv = argv;
    this->canHaveSource = canHaveSource;

    this->sourceNode = 0;

    hasInput = false;
    hasSourceNode = false;
    Parse();
}

bool ArgumentParser::Parse() {
    try {
        for (int i = 1; i < argc - 1; i = i + 2) {
            //argv[i]

            if (strcmp(argv[i], "--input") == 0) {
                input = string(argv[i + 1]);
                hasInput = true;
            } else if (strcmp(argv[i], "--source") == 0 && canHaveSource) {
                sourceNode = atoi(argv[i + 1]);
                hasSourceNode = true;
            } else if (strcmp(argv[i], "--method") == 0) {
                method = atoi(argv[i + 1]);
            } else if (strcmp(argv[i], "--adviseK") == 0) {
                adviseK = atof(argv[i + 1]);
            }
        }

        if (hasInput)
            return true;
    }
    catch (const std::exception &strException) {
        std::cerr << strException.what() << "\n";
        GenerateHelpString();
        exit(0);
    }
    catch (...) {
        std::cerr << "An exception has occurred.\n";
        GenerateHelpString();
        exit(0);
    }
}

string ArgumentParser::GenerateHelpString() {
    string str = "\nRequired arguments:";
    str += "\n    [--input]: Input graph file. E.g., --input FacebookGraph.txt";
    str += "\nOptional arguments";
    if (canHaveSource)
        str += "\n    [--source]:  Begins from the source (Default: 0). E.g., --source 10";
    return str;
}
*/
