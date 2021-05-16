//
// Created by gxl on 2020/12/31.
//
#include <utility>

#include "pagerank.cuh"

void prOpt(string prPath, float adviseK) {
    cout << "===============prOpt==============" << endl;
    ulong edgeIterationSum = 0;
    ulong edgeIterationMax = 0;
    uint testNumNodes = 0;
    ulong testNumEdge = 0;
    ulong transferSum = 0;
    uint *nodePointersI;
    uint *edgeList;
    uint *outDegree;
    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(prPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(uint));
    uint numEdge = 0;
    infile.read((char *) &numEdge, sizeof(uint));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;
    outDegree = new uint[testNumNodes];
    infile.read((char *) outDegree, sizeof(uint) * testNumNodes);
    nodePointersI = new uint[testNumNodes];
    infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
    edgeList = new uint[testNumEdge];
    infile.read((char *) edgeList, sizeof(uint) * testNumEdge);
    infile.close();
    unsigned long max_partition_size;
    unsigned long total_gpu_size;

    auto start = std::chrono::steady_clock::now();
    auto startPreCaculate = std::chrono::steady_clock::now();
    //CPU
    long durationRead;
    uint maxStaticNode = 0;
    uint *degree;
    float *value;
    uint *label;
    bool *isInStatic;
    uint *overloadNodeList;
    uint *staticNodePointer;
    uint *activeNodeList;
    uint *activeOverloadNodePointers;
    vector<PartEdgeListInfo> partEdgeListInfoArr;
    /*
     * overloadEdgeList overload edge list in every iteration
     * */
    uint *overloadEdgeList;
    bool isFromTail = false;
    //GPU
    uint *staticEdgeListD;
    uint *overloadEdgeListD;
    bool *isInStaticD;
    uint *overloadNodeListD;
    uint *staticNodePointerD;
    uint *degreeD;
    uint *outDegreeD;
    // async need two labels
    uint *isActiveD;
    uint *isStaticActive;
    uint *isOverloadActive;
    float *valueD;
    float *sumD;
    uint *activeNodeListD;
    uint *activeNodeLabelingPrefixD;
    uint *overloadLabelingPrefixD;
    uint *activeOverloadNodePointersD;
    uint *activeOverloadDegreeD;

    degree = new uint[testNumNodes];
    value = new float[testNumNodes];
    label = new uint[testNumNodes];
    isInStatic = new bool[testNumNodes];
    overloadNodeList = new uint[testNumNodes];
    staticNodePointer = new uint[testNumNodes];
    activeNodeList = new uint[testNumNodes];
    activeOverloadNodePointers = new uint[testNumNodes];

    //getMaxPartitionSize(max_partition_size, testNumNodes);
    getMaxPartitionSize(max_partition_size, total_gpu_size, testNumNodes, adviseK, sizeof(uint), testNumEdge, 15);
    for (uint i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];
    memcpy(staticNodePointer, nodePointersI, testNumNodes * sizeof(uint));

    gpuErrorcheck(cudaMalloc(&staticEdgeListD, max_partition_size * sizeof(uint)));
    auto startmove = std::chrono::steady_clock::now();
    gpuErrorcheck(
            cudaMemcpy(staticEdgeListD, edgeList, max_partition_size * sizeof(uint), cudaMemcpyHostToDevice));
    auto endMove = std::chrono::steady_clock::now();
    long testDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endMove - startmove).count();
    cout << "move duration " << testDuration << endl;
    gpuErrorcheck(cudaMalloc(&isInStaticD, testNumNodes * sizeof(bool)))
    gpuErrorcheck(cudaMalloc(&overloadNodeListD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&staticNodePointerD, testNumNodes * sizeof(uint)))
    gpuErrorcheck(cudaMemcpy(staticNodePointerD, nodePointersI, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));

    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = 1;
        value[i] = 1.0;

        if (nodePointersI[i] < max_partition_size && (nodePointersI[i] + degree[i] - 1) < max_partition_size) {
            isInStatic[i] = true;
            if (i > maxStaticNode) maxStaticNode = i;
        } else {
            isInStatic[i] = false;
        }
    }
    cudaMemcpy(isInStaticD, isInStatic, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
    uint partOverloadSize = total_gpu_size - max_partition_size;
    uint overloadSize = maxStaticNode > 0 ? testNumEdge - nodePointersI[maxStaticNode + 1] : testNumEdge;
    cout << "max_partition_size: " << max_partition_size << "  maxStaticNode: " << maxStaticNode << endl;
    //uint partOverloadSize = max_partition_size / 2;
    cout << " partOverloadSize " << partOverloadSize << " overloadSize " << overloadSize << endl;
    overloadEdgeList = (uint *) malloc(overloadSize * sizeof(uint));
    if (overloadEdgeList == nullptr) {
        cout << "overloadEdgeList is null" << endl;
        return;
    }
    gpuErrorcheck(cudaMalloc(&overloadEdgeListD, partOverloadSize * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&degreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&outDegreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&isActiveD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&isStaticActive, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&isOverloadActive, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&valueD, testNumNodes * sizeof(float)));
    gpuErrorcheck(cudaMalloc(&sumD, testNumNodes * sizeof(float)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&overloadLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeListD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadNodePointersD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadDegreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(degreeD, degree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(outDegreeD, outDegree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemset(sumD, 0.0f, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(isActiveD, label, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemset(isStaticActive, 0, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemset(isOverloadActive, 0, testNumNodes * sizeof(uint)));

    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    //setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(isActiveD);
    thrust::device_ptr<unsigned int> ptr_labeling_static(isStaticActive);
    thrust::device_ptr<unsigned int> ptr_labeling_overload(isOverloadActive);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    thrust::device_ptr<unsigned int> ptrOverloadDegree(activeOverloadDegreeD);
    thrust::device_ptr<unsigned int> ptrOverloadPrefixsum(overloadLabelingPrefixD);

    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    uint nodeSum = activeNodesNum;
    ulong overloadEdgeSum = 0;
    auto startCpu = std::chrono::steady_clock::now();
    auto endReadCpu = std::chrono::steady_clock::now();
    long durationReadCpu = 0;

    auto startSwap = std::chrono::steady_clock::now();
    auto endSwap = std::chrono::steady_clock::now();
    long durationSwap = 0;

    auto startGpuProcessing = std::chrono::steady_clock::now();
    auto endGpuProcessing = std::chrono::steady_clock::now();
    long durationGpuProcessing = 0;

    auto startOverloadGpuProcessing = std::chrono::steady_clock::now();
    auto endOverloadGpuProcessing = std::chrono::steady_clock::now();
    long durationOverloadGpuProcessing = 0;

    auto startDynamicMemoryCpy = std::chrono::steady_clock::now();
    auto endDynamicMemoryCpy = std::chrono::steady_clock::now();
    long durationDynamicMemoryCpy = 0;

    auto startDynamic = std::chrono::steady_clock::now();
    auto endDynamic = std::chrono::steady_clock::now();
    long durationDynamic = 0;

    auto startPreGpuProcessing = std::chrono::steady_clock::now();
    auto endPreGpuProcessing = std::chrono::steady_clock::now();
    long durationPreGpuProcessing = 0;
    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    //cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;
    cudaStream_t steamStatic, streamDynamic;
    cudaStreamCreate(&steamStatic);
    cudaStreamCreate(&streamDynamic);

    //uint cursorStartSwap = staticFragmentNum + 1;
    uint swapValidNodeSum = 0;
    uint swapValidEdgeSum = 0;
    uint swapNotValidNodeSum = 0;
    uint swapNotValidEdgeSum = 0;
    uint visitEdgeSum = 0;
    uint swapInEdgeSum = 0;
    uint partOverloadSum = 0;
    /*uint coutTemp = 0;
    for (uint i = 0; i < testNumNodes; i ++) {
        coutTemp += label[i];
    }
    cout << "coutTemp " << coutTemp << endl;*/

    long TIME = 0;
    for (int testIndex = 0; testIndex < 1; testIndex++) {

        for (uint i = 0; i < testNumNodes; i++) {
            label[i] = 1;
            value[i] = 1.0;

            if (nodePointersI[i] < max_partition_size && (nodePointersI[i] + degree[i] - 1) < max_partition_size) {
                isInStatic[i] = true;
                if (i > maxStaticNode) maxStaticNode = i;
            } else {
                isInStatic[i] = false;
            }
        }
        cudaMemcpy(isInStaticD, isInStatic, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
        gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemset(sumD, 0.0f, testNumNodes * sizeof(uint)));
        gpuErrorcheck(cudaMemcpy(isActiveD, label, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemset(isStaticActive, 0, testNumNodes * sizeof(uint)));
        gpuErrorcheck(cudaMemset(isOverloadActive, 0, testNumNodes * sizeof(uint)));
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);

        auto startProcessing = std::chrono::steady_clock::now();
        while (activeNodesNum > 2) {
            startPreGpuProcessing = std::chrono::steady_clock::now();
            iter++;
            /*if (iter == 2) {
                break;
            }*/
            //cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;
            setStaticAndOverloadLabel<<<grid, block>>>(testNumNodes, isActiveD, isStaticActive, isOverloadActive,
                                                       isInStaticD);
            uint staticNodeNum = thrust::reduce(ptr_labeling_static, ptr_labeling_static + testNumNodes);
            if (staticNodeNum > 0) {
                //cout << "iter " << iter << " staticNodeNum " << staticNodeNum << endl;
                thrust::exclusive_scan(ptr_labeling_static, ptr_labeling_static + testNumNodes, ptr_labeling_prefixsum);
                setStaticActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeListD, isStaticActive,
                                                          activeNodeLabelingPrefixD);
            }

            uint overloadNodeNum = thrust::reduce(ptr_labeling_overload, ptr_labeling_overload + testNumNodes);
            uint overloadEdgeNum = 0;
            if (overloadNodeNum > 0) {
                //cout << "iter " << iter << " overloadNodeNum " << overloadNodeNum << endl;
                thrust::exclusive_scan(ptr_labeling_overload, ptr_labeling_overload + testNumNodes,
                                       ptrOverloadPrefixsum);
                setOverloadNodePointerSwap<<<grid, block>>>(testNumNodes, overloadNodeListD, activeOverloadDegreeD,
                                                            isOverloadActive,
                                                            overloadLabelingPrefixD, degreeD);
                thrust::exclusive_scan(ptrOverloadDegree, ptrOverloadDegree + overloadNodeNum,
                                       activeOverloadNodePointersD);
                overloadEdgeNum = thrust::reduce(thrust::device, ptrOverloadDegree,
                                                 ptrOverloadDegree + overloadNodeNum, 0);
                //cout << "iter " << iter << " overloadEdgeNum " << overloadEdgeNum << endl;
                long tempSum = (long) overloadEdgeNum * 4;
                cout << "iter " << iter << " overloadEdgeNum " << tempSum << endl;
                overloadEdgeSum += overloadEdgeNum;
                if (overloadEdgeNum > edgeIterationMax) {
                    edgeIterationMax = overloadEdgeNum;
                }
            }
            endPreGpuProcessing = std::chrono::steady_clock::now();
            durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endPreGpuProcessing - startPreGpuProcessing).count();
            startGpuProcessing = std::chrono::steady_clock::now();
            prSumKernel_static<<<grid, block, 0, steamStatic>>>(staticNodeNum, activeNodeListD,
                                                                staticNodePointerD,
                                                                staticEdgeListD, degreeD, outDegreeD, valueD, sumD);
            //cudaDeviceSynchronize();
            if (overloadNodeNum > 0) {
                startCpu = std::chrono::steady_clock::now();
                cudaMemcpyAsync(overloadNodeList, overloadNodeListD, overloadNodeNum * sizeof(uint),
                                cudaMemcpyDeviceToHost,
                                streamDynamic);
                cudaMemcpyAsync(activeOverloadNodePointers, activeOverloadNodePointersD, overloadNodeNum * sizeof(uint),
                                cudaMemcpyDeviceToHost, streamDynamic);
                int threadNum = 20;
                if (overloadNodeNum < 50) {
                    threadNum = 1;
                }
                thread runThreads[threadNum];

                for (int i = 0; i < threadNum; i++) {
                    runThreads[i] = thread(fillDynamic,
                                           i,
                                           threadNum,
                                           0,
                                           overloadNodeNum,
                                           degree,
                                           activeOverloadNodePointers,
                                           nodePointersI,
                                           overloadNodeList,
                                           overloadEdgeList,
                                           edgeList);
                }

                for (unsigned int t = 0; t < threadNum; t++) {
                    runThreads[t].join();
                }
                caculatePartInfoForEdgeList(activeOverloadNodePointers, overloadNodeList, degree, partEdgeListInfoArr,
                                            overloadNodeNum, partOverloadSize, overloadEdgeNum);

                endReadCpu = std::chrono::steady_clock::now();
                durationReadCpu += std::chrono::duration_cast<std::chrono::milliseconds>(endReadCpu - startCpu).count();
                cudaDeviceSynchronize();
                //gpuErrorcheck(cudaPeekAtLastError())
                endGpuProcessing = std::chrono::steady_clock::now();
                durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                        endGpuProcessing - startGpuProcessing).count();
                partOverloadSum += partEdgeListInfoArr.size();
                for (uint i = 0; i < partEdgeListInfoArr.size(); i++) {
                    startDynamic = std::chrono::steady_clock::now();
                    startDynamicMemoryCpy = std::chrono::steady_clock::now();
                    gpuErrorcheck(cudaMemcpy(overloadEdgeListD, overloadEdgeList +
                                                                activeOverloadNodePointers[partEdgeListInfoArr[i].partStartIndex],
                                             partEdgeListInfoArr[i].partEdgeNums * sizeof(uint),
                                             cudaMemcpyHostToDevice))
                    transferSum += partEdgeListInfoArr[i].partEdgeNums;
                    cudaDeviceSynchronize();
                    endDynamicMemoryCpy = std::chrono::steady_clock::now();
                    durationDynamicMemoryCpy += std::chrono::duration_cast<std::chrono::milliseconds>(
                            endDynamicMemoryCpy - startDynamicMemoryCpy).count();
                    startOverloadGpuProcessing = std::chrono::steady_clock::now();
                    prSumKernel_dynamic<<<grid, block, 0, streamDynamic>>>(partEdgeListInfoArr[i].partStartIndex,
                                                                           partEdgeListInfoArr[i].partActiveNodeNums,
                                                                           overloadNodeListD,
                                                                           activeOverloadNodePointersD,
                                                                           overloadEdgeListD, degreeD, outDegreeD,
                                                                           valueD,
                                                                           sumD);
                    cudaDeviceSynchronize();
                    gpuErrorcheck(cudaPeekAtLastError())
                    endOverloadGpuProcessing = std::chrono::steady_clock::now();
                    durationOverloadGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                            endOverloadGpuProcessing - startOverloadGpuProcessing).count();
                    endDynamic = std::chrono::steady_clock::now();
                    durationDynamic += std::chrono::duration_cast<std::chrono::milliseconds>(
                            endDynamic - startDynamic).count();
                }
                //gpuErrorcheck(cudaPeekAtLastError())
            } else {
                cudaDeviceSynchronize();
                endGpuProcessing = std::chrono::steady_clock::now();
                durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                        endGpuProcessing - startGpuProcessing).count();
            }
            startPreGpuProcessing = std::chrono::steady_clock::now();
            prKernel_Opt<<<grid, block>>>(testNumNodes, valueD, sumD, isActiveD);
            cudaDeviceSynchronize();
            activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
            nodeSum += activeNodesNum;
            endPreGpuProcessing = std::chrono::steady_clock::now();
            durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endPreGpuProcessing - startPreGpuProcessing).count();
        }
        auto endRead = std::chrono::steady_clock::now();
        durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
        /*cudaMemcpy(value, valueD, testNumNodes * sizeof(uint), cudaMemcpyDeviceToHost);
        for (int i = 1; i < 31; i++) {
            cout << i << " : " << value[i] << endl;
        }*/
        transferSum += max_partition_size * sizeof(uint);
        cout << "iterationSum " << iter << endl;
        double edgeIterationAvg = (double) overloadEdgeSum / (double) testNumEdge / iter;
        double edgeIterationMaxAvg = (double) edgeIterationMax / (double) testNumEdge;
        cout << "edgeIterationAvg " << edgeIterationAvg << " edgeIterationMaxAvg " << edgeIterationMaxAvg << endl;
        cout << "transferSum : " << 4 * transferSum << " byte" << endl;
        cout << "finish time : " << durationRead << " ms" << endl;
        cout << "total time : " << testDuration + durationRead << " ms" << endl;
        cout << "cpu time : " << durationReadCpu << " ms" << endl;
        cout << "fact processing time : " << durationGpuProcessing << " ms" << endl;
        cout << "durationOverloadGpuProcessing time : " << durationOverloadGpuProcessing << " ms" << endl;
        cout << "durationDynamicMemoryCpy time : " << durationDynamicMemoryCpy << " ms" << endl;
        cout << "durationDynamictime : " << durationDynamic << " ms" << endl;

        cout << "gpu pre processing time : " << durationPreGpuProcessing << " ms" << endl;
        //cout << "overloadEdgeSum : " << overloadEdgeSum << " " << endl;
        cout << "partOverloadSum : " << partOverloadSum << " " << endl;
        //cout << "nodeSum: " << nodeSum << endl;
        TIME += durationRead;
    }
    cout << "TIME " << (float) TIME / (float) 10;
    cudaFree(staticEdgeListD);
    //cudaFree(edgeListOverloadManage);
    cudaFree(degreeD);
    cudaFree(isActiveD);
    cudaFree(valueD);
    cudaFree(activeNodeListD);
    cudaFree(activeNodeLabelingPrefixD);
    cudaFree(activeOverloadNodePointersD);
    cudaFree(activeOverloadDegreeD);
    cudaFree(isInStaticD);
    cudaFree(staticNodePointerD);
    cudaFree(overloadNodeListD);
    cudaFree(overloadEdgeListD);
    cudaFree(isStaticActive);
    cudaFree(isOverloadActive);
    cudaFree(overloadLabelingPrefixD);
    delete[]            label;
    delete[]            degree;
    delete[]            value;
    delete[]            activeNodeList;
    delete[]            activeOverloadNodePointers;
    delete[] isInStatic;
    delete[] overloadNodeList;
    delete[] staticNodePointer;
    delete[] overloadEdgeList;
    partEdgeListInfoArr.clear();
};

void prOptSwap() {
    cout << "===============prOptSwap==============" << endl;
    uint testNumNodes = 0;
    ulong testNumEdge = 0;
    ulong transferSum = 0;
    uint *nodePointersI;
    uint *edgeList;
    uint *outDegree;
    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(prGraphPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(uint));
    uint numEdge = 0;
    infile.read((char *) &numEdge, sizeof(uint));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;
    outDegree = new uint[testNumNodes];
    infile.read((char *) outDegree, sizeof(uint) * testNumNodes);
    nodePointersI = new uint[testNumNodes];
    infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
    edgeList = new uint[testNumEdge];
    infile.read((char *) edgeList, sizeof(uint) * testNumEdge);
    infile.close();
    unsigned long max_partition_size;
    unsigned long total_gpu_size;

    auto start = std::chrono::steady_clock::now();
    auto startPreCaculate = std::chrono::steady_clock::now();
    //CPU
    long durationRead;
    uint fragmentNum = testNumEdge / fragment_size;
    uint staticFragmentNum;
    uint maxStaticNode = 0;
    uint *degree;
    float *value;
    uint *label;
    uint *staticFragmentToNormalMap;
    bool *isInStatic;
    uint *overloadNodeList;
    uint *staticNodePointer;
    uint *staticFragmentData;
    uint *overloadFragmentData;
    uint *activeNodeList;
    uint *activeOverloadNodePointers;
    vector<PartEdgeListInfo> partEdgeListInfoArr;
    uint *fragmentNormalMapToStatic;
    /*
     * overloadEdgeList overload edge list in every iteration
     * */
    uint *overloadEdgeList;
    FragmentData *fragmentData;
    bool *isFragmentActive;
    bool isFromTail = false;
    //GPU
    uint *staticEdgeListD;
    uint *overloadEdgeListD;
    bool *isInStaticD;
    uint *overloadNodeListD;
    uint *staticNodePointerD;
    uint *fragmentNormalMapToStaticD;
    uint *nodePointerD;
    uint *fragmentVisitRecordsD;
    uint *staticFragmentDataD;
    uint *canSwapStaticFragmentDataD;
    uint *canSwapFragmentPrefixSumD;
    uint *degreeD;
    uint *outDegreeD;
    uint *isActiveD;
    uint *isStaticActive;
    uint *isOverloadActive;
    float *valueD;
    float *sumD;
    uint *activeNodeListD;
    uint *activeNodeLabelingPrefixD;
    uint *overloadLabelingPrefixD;
    uint *activeOverloadNodePointersD;
    uint *activeOverloadDegreeD;
    uint *staticFragmentRecordD;
    bool *isFragmentActiveD;

    degree = new uint[testNumNodes];
    value = new float[testNumNodes];
    label = new uint[testNumNodes];
    isInStatic = new bool[testNumNodes];
    overloadNodeList = new uint[testNumNodes];
    staticNodePointer = new uint[testNumNodes];
    activeNodeList = new uint[testNumNodes];
    activeOverloadNodePointers = new uint[testNumNodes];
    fragmentData = new FragmentData[fragmentNum];
    isFragmentActive = new bool[fragmentNum];
    fragmentNormalMapToStatic = new uint[fragmentNum];

    //getMaxPartitionSize(max_partition_size, testNumNodes);
    getMaxPartitionSize(max_partition_size, total_gpu_size, testNumNodes, 0, sizeof(uint), testNumEdge, 16);
    staticFragmentNum = max_partition_size / fragment_size;
    staticFragmentToNormalMap = new uint[staticFragmentNum];
    staticFragmentData = new uint[staticFragmentNum];
    overloadFragmentData = new uint[fragmentNum];
    for (uint i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];
    memcpy(staticNodePointer, nodePointersI, testNumNodes * sizeof(uint));

    gpuErrorcheck(cudaMalloc(&staticEdgeListD, max_partition_size * sizeof(uint)));
    gpuErrorcheck(
            cudaMemcpy(staticEdgeListD, edgeList, max_partition_size * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMalloc(&isInStaticD, testNumNodes * sizeof(bool)))
    gpuErrorcheck(cudaMalloc(&overloadNodeListD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&staticNodePointerD, testNumNodes * sizeof(uint)))
    gpuErrorcheck(cudaMemcpy(staticNodePointerD, nodePointersI, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMalloc(&nodePointerD, testNumNodes * sizeof(uint)))
    gpuErrorcheck(cudaMemcpy(nodePointerD, nodePointersI, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));

    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = true;
        value[i] = 1.0;

        uint pointStartFragmentIndex = nodePointersI[i] / fragment_size;
        uint pointEndFragmentIndex =
                degree[i] == 0 ? pointStartFragmentIndex : (nodePointersI[i] + degree[i] - 1) / fragment_size;
        if (pointStartFragmentIndex == pointEndFragmentIndex && pointStartFragmentIndex >= 0 &&
            pointStartFragmentIndex < fragmentNum) {
            if (fragmentData[pointStartFragmentIndex].vertexNum == 0) {
                fragmentData[pointStartFragmentIndex].startVertex = i;
            } else if (fragmentData[pointStartFragmentIndex].startVertex > i) {
                fragmentData[pointStartFragmentIndex].startVertex = i;
            }
            fragmentData[pointStartFragmentIndex].vertexNum++;
        }

        if (nodePointersI[i] < max_partition_size && (nodePointersI[i] + degree[i] - 1) < max_partition_size) {
            isInStatic[i] = true;
            if (i > maxStaticNode) maxStaticNode = i;
        } else {
            isInStatic[i] = false;
        }
    }
    cudaMemcpy(isInStaticD, isInStatic, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
    cout << "max_partition_size: " << max_partition_size << "  maxStaticNode: " << maxStaticNode << endl;
    cout << "fragmentNum " << fragmentNum << " staticFragmentNum " << staticFragmentNum << endl;
    for (int i = 0; i < staticFragmentNum; i++) {
        fragmentData[i].isIn = true;
    }
    for (uint i = 0; i < staticFragmentNum; i++) {
        staticFragmentToNormalMap[i] = i;
    }
    for (uint i = 0; i < fragmentNum; i++) {
        isFragmentActive[i] = true;
        fragmentNormalMapToStatic[i] = i;
    }
    //uint partOverloadSize = max_partition_size / 2;
    uint partOverloadSize = total_gpu_size - max_partition_size;
    uint overloadSize = testNumEdge - nodePointersI[maxStaticNode + 1];
    cout << " partOverloadSize " << partOverloadSize << " overloadSize " << overloadSize << endl;
    overloadEdgeList = (uint *) malloc(overloadSize * sizeof(uint));
    if (overloadEdgeList == NULL) {
        cout << "overloadEdgeList is null" << endl;
        return;
    }
    gpuErrorcheck(cudaMalloc(&overloadEdgeListD, partOverloadSize * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&isFragmentActiveD, fragmentNum * sizeof(bool)));
    gpuErrorcheck(cudaMemcpy(isFragmentActiveD, isFragmentActive, fragmentNum * sizeof(bool), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMalloc(&staticFragmentDataD, staticFragmentNum * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&fragmentVisitRecordsD, fragmentNum * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&fragmentNormalMapToStaticD, fragmentNum * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(fragmentNormalMapToStaticD, fragmentNormalMapToStatic, fragmentNum * sizeof(uint),
                             cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMalloc(&canSwapStaticFragmentDataD, staticFragmentNum * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&canSwapFragmentPrefixSumD, staticFragmentNum * sizeof(uint)));
    thrust::device_ptr<unsigned int> ptr_canSwapFragment(canSwapStaticFragmentDataD);
    thrust::device_ptr<unsigned int> ptr_canSwapFragmentPrefixSum(canSwapFragmentPrefixSumD);
    gpuErrorcheck(cudaMalloc(&degreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&outDegreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&isActiveD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&isStaticActive, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&isOverloadActive, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&valueD, testNumNodes * sizeof(float)));
    gpuErrorcheck(cudaMalloc(&sumD, testNumNodes * sizeof(float)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&overloadLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeListD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadNodePointersD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadDegreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(degreeD, degree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(outDegreeD, outDegree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemset(sumD, 0.0f, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(isActiveD, label, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemset(isStaticActive, 0, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemset(isOverloadActive, 0, testNumNodes * sizeof(uint)));

    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    //setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(isActiveD);
    thrust::device_ptr<unsigned int> ptr_labeling_static(isStaticActive);
    thrust::device_ptr<unsigned int> ptr_labeling_overload(isOverloadActive);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    thrust::device_ptr<unsigned int> ptrOverloadDegree(activeOverloadDegreeD);
    thrust::device_ptr<unsigned int> ptrOverloadPrefixsum(overloadLabelingPrefixD);

    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    uint nodeSum = activeNodesNum;
    uint overloadEdgeSum = 0;
    auto startCpu = std::chrono::steady_clock::now();
    auto endReadCpu = std::chrono::steady_clock::now();
    long durationReadCpu = 0;

    auto startSwap = std::chrono::steady_clock::now();
    auto endSwap = std::chrono::steady_clock::now();
    long durationSwap = 0;

    auto startGpuProcessing = std::chrono::steady_clock::now();
    auto endGpuProcessing = std::chrono::steady_clock::now();
    long durationGpuProcessing = 0;

    auto startOverloadGpuProcessing = std::chrono::steady_clock::now();
    auto endOverloadGpuProcessing = std::chrono::steady_clock::now();
    long durationOverloadGpuProcessing = 0;

    auto startPreGpuProcessing = std::chrono::steady_clock::now();
    auto endPreGpuProcessing = std::chrono::steady_clock::now();
    long durationPreGpuProcessing = 0;
    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;
    cudaStream_t steamStatic, steamDynamic;
    cudaStreamCreate(&steamStatic);
    cudaStreamCreate(&steamDynamic);
    auto startMemoryTraverse = std::chrono::steady_clock::now();
    auto endMemoryTraverse = std::chrono::steady_clock::now();
    long durationMemoryTraverse = 0;
    auto startProcessing = std::chrono::steady_clock::now();
    uint cursorStartSwap = isFromTail ? fragmentNum - 1 : staticFragmentNum + 1;
    //uint cursorStartSwap = staticFragmentNum + 1;
    uint swapValidNodeSum = 0;
    uint swapValidEdgeSum = 0;
    uint swapNotValidNodeSum = 0;
    uint swapNotValidEdgeSum = 0;
    uint visitEdgeSum = 0;
    ulong swapInEdgeSum = 0;
    uint partOverloadSum = 0;
    /*uint moveInSum = 0;
    uint moveOutSum = 0;*/
    while (activeNodesNum > 0) {
        startPreGpuProcessing = std::chrono::steady_clock::now();
        iter++;
        //cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;
        setStaticAndOverloadLabel4Pr<<<grid, block>>>(testNumNodes, isActiveD, isStaticActive, isOverloadActive,
                                                      isInStaticD, fragmentVisitRecordsD, nodePointerD, fragment_size,
                                                      degreeD, isFragmentActiveD);
        uint staticNodeNum = thrust::reduce(ptr_labeling_static, ptr_labeling_static + testNumNodes);
        if (staticNodeNum > 0) {
            //cout << "iter " << iter << " staticNodeNum " << staticNodeNum << endl;
            thrust::exclusive_scan(ptr_labeling_static, ptr_labeling_static + testNumNodes, ptr_labeling_prefixsum);
            setStaticActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeListD, isStaticActive,
                                                      activeNodeLabelingPrefixD);
        }

        uint overloadNodeNum = thrust::reduce(ptr_labeling_overload, ptr_labeling_overload + testNumNodes);
        uint overloadEdgeNum = 0;
        if (overloadNodeNum > 0) {
            //cout << "iter " << iter << " overloadNodeNum " << overloadNodeNum << endl;
            thrust::exclusive_scan(ptr_labeling_overload, ptr_labeling_overload + testNumNodes, ptrOverloadPrefixsum);
            setOverloadNodePointerSwap<<<grid, block>>>(testNumNodes, overloadNodeListD, activeOverloadDegreeD,
                                                        isOverloadActive,
                                                        overloadLabelingPrefixD, degreeD);
            thrust::exclusive_scan(ptrOverloadDegree, ptrOverloadDegree + overloadNodeNum, activeOverloadNodePointersD);
            overloadEdgeNum = thrust::reduce(thrust::device, ptrOverloadDegree,
                                             ptrOverloadDegree + overloadNodeNum, 0);
            //cout << "iter " << iter << " overloadEdgeNum " << overloadEdgeNum << endl;
            overloadEdgeSum += overloadEdgeNum;

        }
        endPreGpuProcessing = std::chrono::steady_clock::now();
        durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endPreGpuProcessing - startPreGpuProcessing).count();
        startGpuProcessing = std::chrono::steady_clock::now();
        prSumKernel_static<<<grid, block, 0, steamStatic>>>(staticNodeNum, activeNodeListD,
                                                            staticNodePointerD,
                                                            staticEdgeListD, degreeD, outDegreeD, valueD, sumD);
        //cudaDeviceSynchronize();
        if (overloadNodeNum > 0) {
            startCpu = std::chrono::steady_clock::now();

            cudaMemcpyAsync(overloadNodeList, overloadNodeListD, overloadNodeNum * sizeof(uint), cudaMemcpyDeviceToHost,
                            steamDynamic);
            cudaMemcpyAsync(activeOverloadNodePointers, activeOverloadNodePointersD, overloadNodeNum * sizeof(uint),
                            cudaMemcpyDeviceToHost, steamDynamic);
            /*cudaMemcpy(staticActiveNodeList, activeNodeListD, staticNodeNum * sizeof(uint), cudaMemcpyDeviceToHost);
            for (uint i = 0;i < staticNodeNum; i++) {
                if (staticActiveNodeList[i] > staticNodeNum) {
                    moveInSum += degree[staticActiveNodeList[i]];
                }
            }
            for (uint i = 0;i < overloadNodeNum; i++) {
                if (overloadNodeList[i] < staticNodeNum) {
                    moveOutSum += degree[overloadNodeList[i]];
                }
            }*/
            int threadNum = 20;
            if (overloadNodeNum < 50) {
                threadNum = 1;
            }
            thread runThreads[threadNum];

            for (int i = 0; i < threadNum; i++) {
                runThreads[i] = thread(fillDynamic,
                                       i,
                                       threadNum,
                                       0,
                                       overloadNodeNum,
                                       degree,
                                       activeOverloadNodePointers,
                                       nodePointersI,
                                       overloadNodeList,
                                       overloadEdgeList,
                                       edgeList);
            }

            for (unsigned int t = 0; t < threadNum; t++) {
                runThreads[t].join();
            }
            caculatePartInfoForEdgeList(activeOverloadNodePointers, overloadNodeList, degree, partEdgeListInfoArr,
                                        overloadNodeNum, partOverloadSize, overloadEdgeNum);

            endReadCpu = std::chrono::steady_clock::now();
            durationReadCpu += std::chrono::duration_cast<std::chrono::milliseconds>(endReadCpu - startCpu).count();
            uint canSwapFragmentNum;
            if (cursorStartSwap < fragmentNum) {
                setFragmentDataOpt4Pr<<<grid, block, 0, steamStatic>>>(canSwapStaticFragmentDataD, fragmentNum,
                                                                       fragmentVisitRecordsD,
                                                                       isFragmentActiveD, fragmentNormalMapToStaticD,
                                                                       staticFragmentNum);

                canSwapFragmentNum = thrust::reduce(ptr_canSwapFragment, ptr_canSwapFragment + staticFragmentNum);
                if (canSwapFragmentNum > 0) {
                    thrust::exclusive_scan(ptr_canSwapFragment, ptr_canSwapFragment + staticFragmentNum,
                                           ptr_canSwapFragmentPrefixSum);
                    setStaticFragmentData<<<grid, block, 0, steamStatic>>>(staticFragmentNum,
                                                                           canSwapStaticFragmentDataD,
                                                                           canSwapFragmentPrefixSumD,
                                                                           staticFragmentDataD);
                    cudaMemcpyAsync(staticFragmentData, staticFragmentDataD, canSwapFragmentNum * sizeof(uint),
                                    cudaMemcpyDeviceToHost, steamStatic);
                    cudaMemcpyAsync(isFragmentActive, isFragmentActiveD, fragmentNum * sizeof(bool),
                                    cudaMemcpyDeviceToHost, steamStatic);
                }
            }

            uint canSwapStaticFragmentIndex = 0;
            cudaDeviceSynchronize();
            //gpuErrorcheck(cudaPeekAtLastError())
            endGpuProcessing = std::chrono::steady_clock::now();
            durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endGpuProcessing - startGpuProcessing).count();
            startMemoryTraverse = std::chrono::steady_clock::now();
            partOverloadSum += partEdgeListInfoArr.size();

            bool needSwap = false;
            for (int i = 0; i < partEdgeListInfoArr.size(); i++) {
                gpuErrorcheck(cudaMemcpy(overloadEdgeListD, overloadEdgeList +
                                                            activeOverloadNodePointers[partEdgeListInfoArr[i].partStartIndex],
                                         partEdgeListInfoArr[i].partEdgeNums * sizeof(uint),
                                         cudaMemcpyHostToDevice))
                transferSum += partEdgeListInfoArr[i].partEdgeNums * sizeof(uint);
                startOverloadGpuProcessing = std::chrono::steady_clock::now();
                prSumKernel_dynamic<<<grid, block, 0, steamDynamic>>>(partEdgeListInfoArr[i].partStartIndex,
                                                                      partEdgeListInfoArr[i].partActiveNodeNums,
                                                                      overloadNodeListD,
                                                                      activeOverloadNodePointersD,
                                                                      overloadEdgeListD, degreeD, outDegreeD, valueD,
                                                                      sumD);
                if (needSwap && canSwapFragmentNum > 0 && cursorStartSwap < fragmentNum) {
                    cout << "iter " << iter << " canSwapFragmentNum " << canSwapFragmentNum << " cursorStartSwap "
                         << cursorStartSwap << endl;
                    startSwap = std::chrono::steady_clock::now();
                    uint swapSum = 0;
                    for (uint i = cursorStartSwap; i < fragmentNum; i++) {
                        if (cudaSuccess == cudaStreamQuery(steamDynamic) ||
                            canSwapStaticFragmentIndex >= canSwapFragmentNum) {
                            if (i < fragmentNum) {
                                cursorStartSwap = i + 1;
                            }
                            canSwapFragmentNum -= swapSum;
                            swapInEdgeSum += swapSum * fragment_size;
                            if (swapSum == 0) {
                                needSwap = false;
                            }
                            endSwap = std::chrono::steady_clock::now();
                            auto moveDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    endSwap - startSwap).count();
                            durationSwap += moveDuration;
                            cout << "iter " << iter << " swapSum " << swapSum << " swap to " << i << endl;
                            cout << "iter " << iter << " swapDuration " << moveDuration << " ms" << endl;
                            break;
                        }
                        if (cudaErrorNotReady == cudaStreamQuery(steamDynamic)) {
                            const FragmentData swapFragmentData = fragmentData[i];
                            if (isFragmentActive[i] && !swapFragmentData.isIn && swapFragmentData.vertexNum > 0) {
                                uint swapStaticFragmentIndex = staticFragmentData[canSwapStaticFragmentIndex++];
                                uint beSwappedFragmentIndex = staticFragmentToNormalMap[swapStaticFragmentIndex];
                                fragmentData[beSwappedFragmentIndex].isVisit = true;
                                fragmentData[beSwappedFragmentIndex].isIn = false;
                                FragmentData beSwappedFragment = fragmentData[beSwappedFragmentIndex];
                                uint moveFrom = testNumEdge;
                                uint moveTo = testNumEdge;
                                uint moveNum = testNumEdge;
                                if (beSwappedFragment.vertexNum > 0 && beSwappedFragmentIndex > 0 &&
                                    beSwappedFragmentIndex < fragmentNum) {

                                    /*if (i == 887550) {
                                        cout << "887550 move in " << endl;
                                    }
                                    if (beSwappedFragmentIndex == 887550) {
                                        cout << "887550 move out " << endl;
                                    }*/
                                    for (uint j = beSwappedFragment.startVertex - 1;
                                         j < beSwappedFragment.startVertex + beSwappedFragment.vertexNum + 1 &&
                                         j < testNumNodes; j++) {
                                        isInStatic[j] = false;
                                    }
                                    for (uint j = swapFragmentData.startVertex;
                                         j < swapFragmentData.startVertex + swapFragmentData.vertexNum; j++) {
                                        isInStatic[j] = true;
                                        staticNodePointer[j] =
                                                nodePointersI[j] - i * fragment_size +
                                                swapStaticFragmentIndex * fragment_size;
                                    }
                                    moveFrom = nodePointersI[swapFragmentData.startVertex];
                                    moveTo = staticNodePointer[swapFragmentData.startVertex];
                                    moveNum = nodePointersI[swapFragmentData.startVertex + swapFragmentData.vertexNum] -
                                              nodePointersI[swapFragmentData.startVertex];

                                    staticFragmentToNormalMap[swapStaticFragmentIndex] = i;
                                    fragmentNormalMapToStatic[i] = swapStaticFragmentIndex;
                                    fragmentData[i].isIn = true;

                                    cudaMemcpyAsync(staticEdgeListD + moveTo, edgeList + moveFrom,
                                                    moveNum * sizeof(uint),
                                                    cudaMemcpyHostToDevice, steamStatic);
                                    cudaMemcpyAsync(isInStaticD + beSwappedFragment.startVertex - 1,
                                                    isInStatic + beSwappedFragment.startVertex - 1,
                                                    (beSwappedFragment.vertexNum + 2) * sizeof(bool),
                                                    cudaMemcpyHostToDevice, steamStatic);
                                    cudaMemcpyAsync(isInStaticD + swapFragmentData.startVertex,
                                                    isInStatic + swapFragmentData.startVertex,
                                                    swapFragmentData.vertexNum * sizeof(bool),
                                                    cudaMemcpyHostToDevice, steamStatic);

                                    cudaMemcpyAsync(staticNodePointerD + swapFragmentData.startVertex,
                                                    staticNodePointer + swapFragmentData.startVertex,
                                                    swapFragmentData.vertexNum * sizeof(uint), cudaMemcpyHostToDevice,
                                                    steamStatic);
                                    cudaMemcpyAsync(fragmentNormalMapToStaticD + i, &swapStaticFragmentIndex,
                                                    sizeof(uint), cudaMemcpyHostToDevice,
                                                    steamStatic);
                                    swapSum++;
                                    continue;
                                }
                            }
                        }
                        if (i == fragmentNum - 1) {
                            needSwap = false;
                        }
                    }
                }
                cudaDeviceSynchronize();
                gpuErrorcheck(cudaPeekAtLastError())
                endOverloadGpuProcessing = std::chrono::steady_clock::now();
                durationOverloadGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                        endOverloadGpuProcessing - startOverloadGpuProcessing).count();
            }
            endMemoryTraverse = std::chrono::steady_clock::now();
            durationMemoryTraverse += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endMemoryTraverse - startMemoryTraverse).count();
            //gpuErrorcheck(cudaPeekAtLastError())
        } else {
            cudaDeviceSynchronize();
            endGpuProcessing = std::chrono::steady_clock::now();
            durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endGpuProcessing - startGpuProcessing).count();
        }

        prKernel_Opt<<<grid, block>>>(testNumNodes, valueD, sumD, isActiveD);
        cudaDeviceSynchronize();
        startPreGpuProcessing = std::chrono::steady_clock::now();
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;
        endPreGpuProcessing = std::chrono::steady_clock::now();
        durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endPreGpuProcessing - startPreGpuProcessing).count();
    }
    //cudaDeviceSynchronize();
    auto endRead = std::chrono::steady_clock::now();
    durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
    cudaMemcpy(value, valueD, testNumNodes * sizeof(uint), cudaMemcpyDeviceToHost);
    for (int i = 1; i < 31; i++) {
        cout << i << " : " << value[i] << endl;
    }
    transferSum += max_partition_size * sizeof(uint);
    cout << "transferSum : " << transferSum << " byte" << endl;
    cout << "finish time : " << durationRead << " ms" << endl;
    cout << "cpu time : " << durationReadCpu << " ms" << endl;
    cout << "fact processing time : " << durationGpuProcessing << " ms" << endl;
    cout << "durationOverloadGpuProcessing time : " << durationOverloadGpuProcessing << " ms" << endl;
    cout << "durationMemoryTraverse time : " << durationMemoryTraverse << " ms" << endl;

    cout << "gpu pre processing time : " << durationPreGpuProcessing << " ms" << endl;
    cout << "overloadEdgeSum : " << overloadEdgeSum << " " << endl;
    cout << "partOverloadSum : " << partOverloadSum << " " << endl;
    cout << "nodeSum: " << nodeSum << endl;
    cout << "move in " << swapInEdgeSum * 4 << " bytes" << endl;
    cout << "move in " << swapInEdgeSum / fragment_size << " parts" << endl;
    cout << "swap dutation " << durationSwap << " parts" << endl;
    /*cout << "in use " << moveInSum * 4 << " bytes" << endl;
    cout << "out use " << moveOutSum * 4 << " bytes" << endl;*/
    cudaFree(staticEdgeListD);
    //cudaFree(edgeListOverloadManage);
    cudaFree(degreeD);
    cudaFree(isActiveD);
    cudaFree(valueD);
    cudaFree(activeNodeListD);
    cudaFree(activeNodeLabelingPrefixD);
    cudaFree(activeOverloadNodePointersD);
    cudaFree(activeOverloadDegreeD);
    cudaFree(isInStaticD);
    cudaFree(staticNodePointerD);
    cudaFree(overloadNodeListD);
    cudaFree(fragmentVisitRecordsD);
    cudaFree(staticFragmentDataD);
    cudaFree(canSwapStaticFragmentDataD);
    cudaFree(canSwapFragmentPrefixSumD);
    cudaFree(overloadEdgeListD);
    cudaFree(isStaticActive);
    cudaFree(isOverloadActive);
    cudaFree(overloadLabelingPrefixD);
    delete[]            label;
    delete[]            degree;
    delete[]            value;
    delete[]            activeNodeList;
    delete[]            activeOverloadNodePointers;
    delete[] staticFragmentData;
    delete[] isInStatic;
    delete[] overloadNodeList;
    delete[] staticNodePointer;
    delete[] staticFragmentToNormalMap;
    delete[] fragmentData;
    delete[] overloadFragmentData;
    delete[] overloadEdgeList;
    partEdgeListInfoArr.clear();
};

void prShareByInDegree(string prPath) {
    cout << "===============prShareByInDegree==============" << endl;
    uint testNumNodes = 0;
    ulong testNumEdge = 0;
    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(prPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(uint));
    uint numEdge = 0;
    infile.read((char *) &numEdge, sizeof(uint));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;
    uint *nodePointersI;
    uint *edgeList;
    uint *outDegree;
    gpuErrorcheck(cudaMallocManaged(&outDegree, (testNumNodes) * sizeof(uint)));
    infile.read((char *) outDegree, sizeof(uint) * testNumNodes);
    gpuErrorcheck(cudaMallocManaged(&nodePointersI, (testNumNodes + 1) * sizeof(uint)));
    infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
    gpuErrorcheck(cudaMallocManaged(&edgeList, (numEdge) * sizeof(uint)));
    infile.read((char *) edgeList, sizeof(uint) * testNumEdge);
    cudaMemAdvise(outDegree, (testNumNodes) * sizeof(uint), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(nodePointersI, (testNumNodes + 1) * sizeof(uint), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(edgeList, (numEdge) * sizeof(uint), cudaMemAdviseSetReadMostly, 0);
    infile.close();
    //preprocessData(nodePointersI, edgeList, testNumNodes, testNumEdge);
    auto endReadGraph = std::chrono::steady_clock::now();
    long durationReadGraph = std::chrono::duration_cast<std::chrono::milliseconds>(
            endReadGraph - startReadGraph).count();
    cout << "read graph time : " << durationReadGraph << "ms" << endl;
    uint *degree;
    cudaMallocManaged(&degree, testNumNodes * sizeof(uint));
    float *value;
    cudaMallocManaged(&value, testNumNodes * sizeof(float));
    int *isActiveNodeList;
    cudaMallocManaged(&isActiveNodeList, testNumNodes * sizeof(int));
    for (uint i = 0; i < testNumNodes; i++) {
        isActiveNodeList[i] = 1;
        value[i] = 1.0f;
        if (i + 1 < testNumNodes) {
            degree[i] = nodePointersI[i + 1] - nodePointersI[i];
        } else {
            degree[i] = testNumEdge - nodePointersI[i];
        }
    }

    thrust::device_ptr<int> ptr_labeling(isActiveNodeList);
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);
    int testTimes = 1;
    for (int i = 0; i < testTimes; i++) {
        int iteration = 0;
        uint activeNodeNum = testNumNodes;
        auto startProcessing = std::chrono::steady_clock::now();
        //vector<vector<uint>> visitRecordByIteration;
        while (activeNodeNum > 2) {
            cout << "iteration " << iteration << " activeNodes " << activeNodeNum << endl;
            prSumKernel_UVM_Out<<<grid, block>>>(testNumNodes, isActiveNodeList, nodePointersI,
                                                 edgeList, degree, outDegree, value);
            cudaDeviceSynchronize();
            //visitRecordByIteration.push_back(countDataByIteration(testNumEdge, testNumNodes, nodePointersI, degree, isActiveNodeList));
            gpuErrorcheck(cudaPeekAtLastError())
            activeNodeNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
            iteration++;
        }
        //writeTrunkVistInIteration(visitRecordByIteration, "./CountByIterationPR.txt");
        auto endRead = std::chrono::steady_clock::now();
        long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
        cout << " finish time : " << durationRead << " ms" << endl;
        for (int i = 0; i < 30; i++) {
            cout << i << " : " << value[i] << endl;
        }
    }
}

void prShare() {
    cout << "===============uvmPR==============" << endl;
    uint testNumNodes = 0;
    ulong testNumEdge = 0;
    uint *nodePointersI;
    uint *edgeList;
    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(testGraphPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(uint));
    uint numEdge = 0;
    infile.read((char *) &numEdge, sizeof(uint));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;
    gpuErrorcheck(cudaMallocManaged(&nodePointersI, (testNumNodes + 1) * sizeof(uint)));
    infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
    gpuErrorcheck(cudaMallocManaged(&edgeList, (numEdge) * sizeof(uint)));
    infile.read((char *) edgeList, sizeof(uint) * testNumEdge);
    cudaMemAdvise(nodePointersI, (testNumNodes + 1) * sizeof(uint), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(edgeList, (numEdge) * sizeof(uint), cudaMemAdviseSetReadMostly, 0);
    infile.close();

    //preprocessData(nodePointersI, edgeList, testNumNodes, testNumEdge);
    auto endReadGraph = std::chrono::steady_clock::now();
    long durationReadGraph = std::chrono::duration_cast<std::chrono::milliseconds>(
            endReadGraph - startReadGraph).count();
    cout << "read graph time : " << durationReadGraph << "ms" << endl;
    uint *degree;
    cudaMallocManaged(&degree, testNumNodes * sizeof(uint));
    float *value;
    cudaMallocManaged(&value, testNumNodes * sizeof(float));
    float *sum;
    cudaMallocManaged(&sum, testNumNodes * sizeof(float));
    int *isActiveNodeList;
    cudaMallocManaged(&isActiveNodeList, testNumNodes * sizeof(int));
    for (uint i = 0; i < testNumNodes; i++) {
        isActiveNodeList[i] = 1;
        value[i] = 1.0f;
        sum[i] = 0.0f;
        if (i + 1 < testNumNodes) {
            degree[i] = nodePointersI[i + 1] - nodePointersI[i];
        } else {
            degree[i] = testNumEdge - nodePointersI[i];
        }
    }
    /*for (uint i = 0; i < testNumNodes; i++) {
        for (uint j = 0; j< degree[i]; j++)
            if (edgeList[nodePointersI[i] + j] == 1) {
                cout << "src " << i << " dest " << 0 << endl;
            }
    }*/
    thrust::device_ptr<int> ptr_labeling(isActiveNodeList);
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);
    int testTimes = 1;
    for (int i = 0; i < testTimes; i++) {
        int iteration = 0;
        uint activeNodeNum = testNumNodes;
        auto startProcessing = std::chrono::steady_clock::now();
        while (activeNodeNum > 0 && iteration < 79) {
            cout << "iteration " << iteration << " activeNodes " << activeNodeNum << endl;
            prSumKernel_UVM<<<grid, block>>>(testNumNodes, isActiveNodeList, nodePointersI,
                                             edgeList, degree, value, sum);
            cudaDeviceSynchronize();
            gpuErrorcheck(cudaPeekAtLastError())

            prKernel_UVM<<<grid, block>>>(testNumNodes, value, sum, isActiveNodeList);
            cudaDeviceSynchronize();
            gpuErrorcheck(cudaPeekAtLastError())
            activeNodeNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
            iteration++;
        }
        auto endRead = std::chrono::steady_clock::now();
        long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
        cout << " finish time : " << durationRead << " ms" << endl;
        for (int i = 0; i < 30; i++) {
            cout << i << " : " << value[i] << endl;
        }
    }
}

void conventionParticipatePR(string prPath) {
    cout << "===============conventionParticipatePR==============" << endl;
    unsigned long transferSum = 0;
    uint testNumNodes = 0;
    ulong testNumEdge = 0;
    uint *nodePointersI;
    uint *edgeList;
    uint *outDegree;
    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(prPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(uint));
    uint numEdge = 0;
    infile.read((char *) &numEdge, sizeof(uint));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;
    outDegree = new uint[testNumNodes];
    infile.read((char *) outDegree, sizeof(uint) * testNumNodes);
    nodePointersI = new uint[testNumNodes];
    infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
    edgeList = new uint[testNumEdge];
    infile.read((char *) edgeList, sizeof(uint) * testNumEdge);
    infile.close();
    unsigned long max_partition_size;
    unsigned long total_gpu_size;
    getMaxPartitionSize(max_partition_size, total_gpu_size, testNumNodes, 0.9, sizeof(uint), 7);
    uint partitionNum;
    if (testNumEdge > max_partition_size) {
        partitionNum = testNumEdge / max_partition_size + 1;
    } else {
        partitionNum = 1;
    }

    uint *inDegree = new uint[testNumNodes];
    auto *value = new float[testNumNodes];
    auto *sum = new float[testNumNodes];
    bool *isActiveNodeList = new bool[testNumNodes];
    CommonPartitionInfo *partitionInfoList = new CommonPartitionInfo[partitionNum];
    bool *needTransferPartition = new bool[partitionNum];
    for (uint i = 0; i < testNumNodes; i++) {
        isActiveNodeList[i] = true;
        value[i] = 1.0f;
        sum[i] = 0.0f;
        if (i + 1 < testNumNodes) {
            inDegree[i] = nodePointersI[i + 1] - nodePointersI[i];
        } else {
            inDegree[i] = testNumEdge - nodePointersI[i];
        }
        if (inDegree[i] > max_partition_size) {
            cout << "node " << i << " degree > maxPartition " << endl;
            return;
        }
    }
    for (uint i = 0; i < partitionNum; i++) {
        partitionInfoList[i].startVertex = -1;
        partitionInfoList[i].endVertex = -1;
        partitionInfoList[i].nodePointerOffset = -1;
        partitionInfoList[i].partitionEdgeSize = -1;
    }
    int tempPartitionIndex = 0;
    uint tempNodeIndex = 0;
    while (tempNodeIndex < testNumNodes) {
        if (partitionInfoList[tempPartitionIndex].startVertex == -1) {
            partitionInfoList[tempPartitionIndex].startVertex = tempNodeIndex;
            partitionInfoList[tempPartitionIndex].endVertex = tempNodeIndex;
            partitionInfoList[tempPartitionIndex].nodePointerOffset = nodePointersI[tempNodeIndex];
            partitionInfoList[tempPartitionIndex].partitionEdgeSize = inDegree[tempNodeIndex];
            tempNodeIndex++;
        } else {
            if (partitionInfoList[tempPartitionIndex].partitionEdgeSize + inDegree[tempNodeIndex] >
                max_partition_size) {
                tempPartitionIndex++;
            } else {
                partitionInfoList[tempPartitionIndex].endVertex = tempNodeIndex;
                partitionInfoList[tempPartitionIndex].partitionEdgeSize += inDegree[tempNodeIndex];
                tempNodeIndex++;
            }
        }
    }

    uint *outDegreeD;
    uint *inDegreeD;
    bool *isActiveNodeListD;
    bool *nextActiveNodeListD;
    uint *nodePointerListD;
    uint *partitionEdgeListD;
    float *valueD;
    float *sumD;

    cudaMalloc(&outDegreeD, testNumNodes * sizeof(uint));
    cudaMalloc(&inDegreeD, testNumNodes * sizeof(uint));
    cudaMalloc(&valueD, testNumNodes * sizeof(float));
    cudaMalloc(&sumD, testNumNodes * sizeof(float));
    cudaMalloc(&isActiveNodeListD, testNumNodes * sizeof(bool));
    cudaMalloc(&nextActiveNodeListD, testNumNodes * sizeof(bool));
    cudaMalloc(&nodePointerListD, testNumNodes * sizeof(uint));
    cudaMalloc(&partitionEdgeListD, max_partition_size * sizeof(uint));

    cudaMemcpy(outDegreeD, outDegree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(inDegreeD, inDegree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(valueD, value, testNumNodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sumD, sum, testNumNodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nodePointerListD, nodePointersI, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemset(nextActiveNodeListD, 0, testNumNodes * sizeof(bool));
    transferSum += 5 * testNumNodes;
    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    int testTimes = 1;
    long timeSum = 0;
    for (int i = 0; i < testTimes; i++) {
        for (int j = 0; j < testNumNodes; j++) {
            isActiveNodeList[j] = true;
        }
        cudaMemcpy(isActiveNodeListD, isActiveNodeList, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
        //uint activeSum = 0;
        int iteration = 0;

        auto startProcessing = std::chrono::steady_clock::now();
        while (true) {
            uint activeNodeNum = 0;
            checkNeedTransferPartitionOpt(needTransferPartition, partitionInfoList, isActiveNodeList, partitionNum,
                                          testNumNodes, activeNodeNum);
            if (activeNodeNum <= 2) {
                break;
            } else {
                //cout << "iteration " << iteration << " activeNodes " << activeNodeNum << endl;
            }
            //cudaMemcpy(isActiveNodeListD, isActiveNodeList, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
            for (int j = 0; j < partitionNum; j++) {
                if (needTransferPartition[j]) {
                    cudaMemcpy(partitionEdgeListD, edgeList + partitionInfoList[j].nodePointerOffset,
                               partitionInfoList[j].partitionEdgeSize * sizeof(uint), cudaMemcpyHostToDevice);
                    transferSum += partitionInfoList[j].partitionEdgeSize;
                    prSumKernel_CommonPartition<<<grid, block>>>(partitionInfoList[j].startVertex,
                                                                 partitionInfoList[j].endVertex,
                                                                 partitionInfoList[j].nodePointerOffset,
                                                                 isActiveNodeListD, nodePointerListD,
                                                                 partitionEdgeListD, inDegreeD, outDegreeD, valueD,
                                                                 sumD);
                    cudaDeviceSynchronize();
                    gpuErrorcheck(cudaPeekAtLastError())
                }
            }
            prKernel_CommonPartition<<<grid, block>>>(testNumNodes, valueD, sumD, isActiveNodeListD);
            cudaDeviceSynchronize();
            gpuErrorcheck(cudaPeekAtLastError())
            cudaMemcpy(isActiveNodeList, isActiveNodeListD, testNumNodes * sizeof(bool), cudaMemcpyDeviceToHost);
            iteration++;
        }
        auto endRead = std::chrono::steady_clock::now();
        long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
        cout << " finish time : " << durationRead << " ms" << endl;
        cudaMemcpy(value, valueD, 30 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 30; i++) {
            cout << i << " : " << value[i] << endl;
        }

        cout << "cpu transfer to gpu " << transferSum * sizeof(uint) << "byte" << endl;
    }

    free(nodePointersI);
    free(edgeList);
    free(inDegree);
    free(isActiveNodeList);
    cudaFree(isActiveNodeListD);
    cudaFree(nextActiveNodeListD);
    cudaFree(nodePointerListD);
    cudaFree(partitionEdgeListD);
    //todo free partitionInfoList needTransferPartition

}

void conventionParticipatePRHalfStatic() {
    cout << "===============conventionParticipatePRHalfStatic==============" << endl;
    unsigned long transferSum = 0;
    uint testNumNodes = 0;
    ulong testNumEdge = 0;
    uint *nodePointersI;
    uint *edgeList;
    uint *outDegree;
    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(prGraphPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(uint));
    uint numEdge = 0;
    infile.read((char *) &numEdge, sizeof(uint));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;
    outDegree = new uint[testNumNodes];
    infile.read((char *) outDegree, sizeof(uint) * testNumNodes);
    nodePointersI = new uint[testNumNodes];
    infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
    edgeList = new uint[testNumEdge];
    infile.read((char *) edgeList, sizeof(uint) * testNumEdge);
    infile.close();
    unsigned long max_partition_size;
    unsigned long total_gpu_size;
    getMaxPartitionSize(max_partition_size, total_gpu_size, testNumNodes, 0.5, sizeof(uint), 7);
    unsigned long static_gpu_size = total_gpu_size - max_partition_size;
    uint partitionNum;
    if (testNumEdge > max_partition_size) {
        partitionNum = testNumEdge / max_partition_size + 1;
    } else {
        partitionNum = 1;
    }

    uint maxStaticNode = 0;
    uint *inDegree = new uint[testNumNodes];
    auto *value = new float[testNumNodes];
    auto *sum = new float[testNumNodes];
    bool *isActiveNodeList = new bool[testNumNodes];
    CommonPartitionInfo *partitionInfoList = new CommonPartitionInfo[partitionNum];
    bool *needTransferPartition = new bool[partitionNum];
    for (uint i = 0; i < testNumNodes; i++) {
        isActiveNodeList[i] = true;
        value[i] = 1.0f;
        sum[i] = 0.0f;
        if (i + 1 < testNumNodes) {
            inDegree[i] = nodePointersI[i + 1] - nodePointersI[i];
        } else {
            inDegree[i] = testNumEdge - nodePointersI[i];
        }
        if (inDegree[i] > max_partition_size) {
            cout << "node " << i << " degree > maxPartition " << endl;
            return;
        }
        if (nodePointersI[i] < static_gpu_size && (nodePointersI[i] + inDegree[i] - 1) < static_gpu_size) {
            if (i > maxStaticNode) maxStaticNode = i;
        }
    }

    for (uint i = 0; i < partitionNum; i++) {
        partitionInfoList[i].startVertex = -1;
        partitionInfoList[i].endVertex = -1;
        partitionInfoList[i].nodePointerOffset = -1;
        partitionInfoList[i].partitionEdgeSize = -1;
    }
    int tempPartitionIndex = 0;
    uint tempNodeIndex = maxStaticNode + 1;
    while (tempNodeIndex < testNumNodes) {
        if (partitionInfoList[tempPartitionIndex].startVertex == -1) {
            partitionInfoList[tempPartitionIndex].startVertex = tempNodeIndex;
            partitionInfoList[tempPartitionIndex].endVertex = tempNodeIndex;
            partitionInfoList[tempPartitionIndex].nodePointerOffset = nodePointersI[tempNodeIndex];
            partitionInfoList[tempPartitionIndex].partitionEdgeSize = inDegree[tempNodeIndex];
            tempNodeIndex++;
        } else {
            if (partitionInfoList[tempPartitionIndex].partitionEdgeSize + inDegree[tempNodeIndex] >
                max_partition_size) {
                tempPartitionIndex++;
            } else {
                partitionInfoList[tempPartitionIndex].endVertex = tempNodeIndex;
                partitionInfoList[tempPartitionIndex].partitionEdgeSize += inDegree[tempNodeIndex];
                tempNodeIndex++;
            }
        }
    }

    uint *outDegreeD;
    uint *inDegreeD;
    bool *isActiveNodeListD;
    bool *nextActiveNodeListD;
    uint *nodePointerListD;
    uint *partitionEdgeListD;
    uint *staticEdgeListD;
    float *valueD;
    float *sumD;

    cudaMalloc(&outDegreeD, testNumNodes * sizeof(uint));
    cudaMalloc(&inDegreeD, testNumNodes * sizeof(uint));
    cudaMalloc(&valueD, testNumNodes * sizeof(float));
    cudaMalloc(&sumD, testNumNodes * sizeof(float));
    cudaMalloc(&isActiveNodeListD, testNumNodes * sizeof(bool));
    cudaMalloc(&nextActiveNodeListD, testNumNodes * sizeof(bool));
    cudaMalloc(&nodePointerListD, testNumNodes * sizeof(uint));
    cudaMalloc(&partitionEdgeListD, max_partition_size * sizeof(uint));
    cudaMalloc(&staticEdgeListD, static_gpu_size * sizeof(uint));

    cudaMemcpy(outDegreeD, outDegree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(inDegreeD, inDegree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(valueD, value, testNumNodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sumD, sum, testNumNodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nodePointerListD, nodePointersI, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemset(nextActiveNodeListD, 0, testNumNodes * sizeof(bool));
    cudaMemcpy(staticEdgeListD, edgeList, static_gpu_size * sizeof(uint), cudaMemcpyHostToDevice);
    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);
    transferSum += 5 * testNumNodes + static_gpu_size;
    int testTimes = 1;
    long timeSum = 0;
    for (int i = 0; i < testTimes; i++) {
        for (int j = 0; j < testNumNodes; j++) {
            isActiveNodeList[j] = true;
        }
        cudaMemcpy(isActiveNodeListD, isActiveNodeList, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
        //uint activeSum = 0;
        int iteration = 0;

        auto startProcessing = std::chrono::steady_clock::now();
        while (true) {
            uint activeNodeNum = 0;
            checkNeedTransferPartition(needTransferPartition, partitionInfoList, isActiveNodeList, partitionNum,
                                       testNumNodes, activeNodeNum);
            if (activeNodeNum <= 0) {
                break;
            } else {
                cout << "iteration " << iteration << " activeNodes " << activeNodeNum << endl;
            }

            prSumKernel_CommonPartition<<<grid, block>>>(0,
                                                         maxStaticNode,
                                                         0,
                                                         isActiveNodeListD, nodePointerListD,
                                                         staticEdgeListD, inDegreeD, outDegreeD, valueD,
                                                         sumD);
            cudaDeviceSynchronize();
            gpuErrorcheck(cudaPeekAtLastError())

            //cudaMemcpy(isActiveNodeListD, isActiveNodeList, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
            for (int j = 0; j < partitionNum; j++) {
                if (needTransferPartition[j]) {
                    cudaMemcpy(partitionEdgeListD, edgeList + partitionInfoList[j].nodePointerOffset,
                               partitionInfoList[j].partitionEdgeSize * sizeof(uint), cudaMemcpyHostToDevice);
                    transferSum += partitionInfoList[j].partitionEdgeSize;
                    prSumKernel_CommonPartition<<<grid, block>>>(partitionInfoList[j].startVertex,
                                                                 partitionInfoList[j].endVertex,
                                                                 partitionInfoList[j].nodePointerOffset,
                                                                 isActiveNodeListD, nodePointerListD,
                                                                 partitionEdgeListD, inDegreeD, outDegreeD, valueD,
                                                                 sumD);
                    cudaDeviceSynchronize();
                    gpuErrorcheck(cudaPeekAtLastError())
                }
            }
            prKernel_CommonPartition<<<grid, block>>>(testNumNodes, valueD, sumD, isActiveNodeListD);
            cudaDeviceSynchronize();
            gpuErrorcheck(cudaPeekAtLastError())
            cudaMemcpy(isActiveNodeList, isActiveNodeListD, testNumNodes * sizeof(bool), cudaMemcpyDeviceToHost);
            iteration++;
        }
        auto endRead = std::chrono::steady_clock::now();
        long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
        cout << " finish time : " << durationRead << " ms" << endl;
        cudaMemcpy(value, valueD, 30 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 30; i++) {
            cout << i << " : " << value[i] << endl;
        }

        cout << "cpu transfer to gpu " << transferSum * sizeof(uint) << "byte" << endl;
    }

    free(nodePointersI);
    free(edgeList);
    free(inDegree);
    free(isActiveNodeList);
    cudaFree(isActiveNodeListD);
    cudaFree(nextActiveNodeListD);
    cudaFree(nodePointerListD);
    cudaFree(partitionEdgeListD);
    //todo free partitionInfoList needTransferPartition

}

void prShareByInDegreeTrace(string prPath) {
    cout << "===============prShareByInDegreeTrace==============" << endl;
    uint testNumNodes = 0;
    ulong testNumEdge = 0;
    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(prPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(uint));
    uint numEdge = 0;
    infile.read((char *) &numEdge, sizeof(uint));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;
    uint *nodePointersI = new uint[testNumNodes];
    uint *outDegree = new uint[testNumNodes];
    uint *edgeList;
    //gpuErrorcheck(cudaMallocManaged(&outDegree, (testNumNodes) * sizeof(uint)));
    infile.read((char *) outDegree, sizeof(uint) * testNumNodes);
    //gpuErrorcheck(cudaMallocManaged(&nodePointersI, (testNumNodes + 1) * sizeof(uint)));
    infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
    gpuErrorcheck(cudaMallocManaged(&edgeList, (numEdge) * sizeof(uint)));
    infile.read((char *) edgeList, sizeof(uint) * testNumEdge);
    //cudaMemAdvise(outDegree, (testNumNodes) * sizeof(uint), cudaMemAdviseSetReadMostly, 0);
    //cudaMemAdvise(nodePointersI, (testNumNodes + 1) * sizeof(uint), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(edgeList, (numEdge) * sizeof(uint), cudaMemAdviseSetReadMostly, 0);
    infile.close();
    //preprocessData(nodePointersI, edgeList, testNumNodes, testNumEdge);
    auto endReadGraph = std::chrono::steady_clock::now();
    long durationReadGraph = std::chrono::duration_cast<std::chrono::milliseconds>(
            endReadGraph - startReadGraph).count();
    cout << "read graph time : " << durationReadGraph << "ms" << endl;
    uint *degree = new uint[testNumNodes];
    //cudaMallocManaged(&degree, testNumNodes * sizeof(uint));
    float *value = new float[testNumNodes];
    //cudaMallocManaged(&value, testNumNodes * sizeof(float));
    int *isActiveNodeList = new int[testNumNodes];
    //cudaMallocManaged(&isActiveNodeList, testNumNodes * sizeof(int));
    for (uint i = 0; i < testNumNodes; i++) {
        isActiveNodeList[i] = 1;
        value[i] = 1.0f;
        if (i + 1 < testNumNodes) {
            degree[i] = nodePointersI[i + 1] - nodePointersI[i];
        } else {
            degree[i] = testNumEdge - nodePointersI[i];
        }
    }
    uint *nodePointersD;
    cudaMalloc(&nodePointersD, testNumNodes * sizeof(uint));
    cudaMemcpy(nodePointersD, nodePointersI, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice);
    uint *outDegreeD;
    cudaMalloc(&outDegreeD, testNumNodes * sizeof(uint));
    cudaMemcpy(outDegreeD, outDegree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice);
    uint *degreeD;
    cudaMalloc(&degreeD, testNumNodes * sizeof(uint));
    cudaMemcpy(degreeD, degree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice);
    float *valueD;
    cudaMalloc(&valueD, testNumNodes * sizeof(float));
    cudaMemcpy(valueD, value, testNumNodes * sizeof(float), cudaMemcpyHostToDevice);
    int *isActiveNodeListD;
    cudaMalloc(&isActiveNodeListD, testNumNodes * sizeof(int));
    cudaMemcpy(isActiveNodeListD, isActiveNodeList, testNumNodes * sizeof(int), cudaMemcpyHostToDevice);

    thrust::device_ptr<int> ptr_labeling(isActiveNodeListD);
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);
    int testTimes = 1;
    for (int i = 0; i < testTimes; i++) {
        int iteration = 0;
        uint activeNodeNum = testNumNodes;
        auto startProcessing = std::chrono::steady_clock::now();
        while (activeNodeNum > 2) {
            /*if (iteration == 2) {
                break;
            }*/
            cout << "iteration " << iteration << " activeNodes " << activeNodeNum << endl;
            prSumKernel_UVM_Out<<<grid, block>>>(testNumNodes, isActiveNodeListD, nodePointersD,
                                                 edgeList, degreeD, outDegreeD, valueD);
            cudaDeviceSynchronize();
            /*cudaMemAdvise(edgeList, testNumEdge * sizeof(uint), cudaMemAdviseSetAccessedBy,
                          cudaCpuDeviceId);
            cudaMemAdvise(edgeList, testNumEdge * sizeof(uint), cudaMemAdviseUnsetAccessedBy,
                          cudaCpuDeviceId);*/
            for (int j = 0; j < testNumEdge; j++) {
                uint temp = edgeList[j];
                uint a = temp + 1;
            }
            gpuErrorcheck(cudaPeekAtLastError())
            activeNodeNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
            iteration++;
        }
        auto endRead = std::chrono::steady_clock::now();
        long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
        cout << " finish time : " << durationRead << " ms" << endl;
        /*for (int i = 0; i < 30; i++) {
            cout << i << " : " << value[i] << endl;
        }*/
    }
}

void prOptRandom(const string &prPath, float adviseK) {
    cout << "===============prOptRandom==============" << endl;
    ulong edgeIterationSum = 0;
    ulong edgeIterationMax = 0;
    uint testNumNodes = 0;
    ulong testNumEdge = 0;
    ulong transferSum = 0;
    uint *nodePointersI;
    uint *edgeList;
    uint *outDegree;
    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(prPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(uint));
    uint numEdge = 0;
    infile.read((char *) &numEdge, sizeof(uint));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;
    outDegree = new uint[testNumNodes];
    infile.read((char *) outDegree, sizeof(uint) * testNumNodes);
    nodePointersI = new uint[testNumNodes];
    infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
    edgeList = new uint[testNumEdge];
    infile.read((char *) edgeList, sizeof(uint) * testNumEdge);
    infile.close();
    unsigned long max_partition_size;
    unsigned long total_gpu_size;

    auto start = std::chrono::steady_clock::now();
    auto startPreCaculate = std::chrono::steady_clock::now();
    //CPU
    long durationRead;
    uint maxStaticNode = 0;
    uint *degree;
    float *value;
    uint *label;
    bool *isInStatic;
    uint *overloadNodeList;
    uint *staticNodePointer;
    uint *activeNodeList;
    uint *activeOverloadNodePointers;
    vector<PartEdgeListInfo> partEdgeListInfoArr;
    uint *overloadEdgeList;
    //GPU
    uint *staticEdgeListD;
    uint *overloadEdgeListD;
    bool *isInStaticD;
    uint *overloadNodeListD;
    uint *staticNodePointerD;
    uint *degreeD;
    uint *outDegreeD;
    // async need two labels
    uint *isActiveD;
    uint *isStaticActive;
    uint *isOverloadActive;
    float *valueD;
    float *sumD;
    uint *activeNodeListD;
    uint *activeNodeLabelingPrefixD;
    uint *overloadLabelingPrefixD;
    uint *activeOverloadNodePointersD;
    uint *activeOverloadDegreeD;

    degree = new uint[testNumNodes];
    value = new float[testNumNodes];
    label = new uint[testNumNodes];
    isInStatic = new bool[testNumNodes];
    overloadNodeList = new uint[testNumNodes];
    staticNodePointer = new uint[testNumNodes];
    activeNodeList = new uint[testNumNodes];
    activeOverloadNodePointers = new uint[testNumNodes];

    getMaxPartitionSize(max_partition_size, total_gpu_size, testNumNodes, adviseK, sizeof(uint), testNumEdge, 15);
    calculateDegree(testNumNodes, nodePointersI, testNumEdge, degree);
    //memcpy(staticNodePointer, nodePointersI, testNumNodes * sizeof(uint));
    uint edgesInStatic = 0;
    float startRate = (1 - (float) max_partition_size / (float) testNumEdge) / 2;
    uint startIndex = (float) testNumNodes * startRate;
    uint tempStaticSum = 0;
    /*for (uint i = testNumNodes - 1; i >= 0; i--) {
        tempStaticSum += degree[i];
        if (tempStaticSum > max_partition_size) {
            startIndex = i;
            break;
        }
    }*/
    startIndex = 0;
    if (nodePointersI[startIndex] + max_partition_size > testNumEdge) {
        startIndex = (float) testNumNodes * 0.1f;
    }
    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = 1;
        value[i] = 1.0;
        if (i >= startIndex && nodePointersI[i] < nodePointersI[startIndex] + max_partition_size - degree[i]) {
            isInStatic[i] = true;
            staticNodePointer[i] = nodePointersI[i] - nodePointersI[startIndex];
            if (i > maxStaticNode) {
                maxStaticNode = i;
            }
            edgesInStatic += degree[i];
        } else {
            isInStatic[i] = false;
        }
    }

    gpuErrorcheck(cudaMalloc(&staticEdgeListD, max_partition_size * sizeof(uint)));
    auto startmove = std::chrono::steady_clock::now();
    gpuErrorcheck(
            cudaMemcpy(staticEdgeListD, edgeList + nodePointersI[startIndex], max_partition_size * sizeof(uint),
                       cudaMemcpyHostToDevice));
    auto endMove = std::chrono::steady_clock::now();
    long testDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endMove - startmove).count();
    cout << "move duration " << testDuration << endl;

    gpuErrorcheck(cudaMalloc(&isInStaticD, testNumNodes * sizeof(bool)))
    gpuErrorcheck(cudaMalloc(&overloadNodeListD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&staticNodePointerD, testNumNodes * sizeof(uint)))
    gpuErrorcheck(
            cudaMemcpy(staticNodePointerD, staticNodePointer, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    cudaMemcpy(isInStaticD, isInStatic, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
    /*cout << "max_partition_size: " << max_partition_size << "  minStaticNode: " << startIndex << "  maxStaticNode: "
         << maxStaticNode << endl;*/
    //uint partOverloadSize = max_partition_size / 2;
    uint partOverloadSize = total_gpu_size - max_partition_size;
    uint overloadSize = testNumEdge - edgesInStatic;
    cout << " partOverloadSize " << partOverloadSize << " overloadSize " << overloadSize << endl;
    overloadEdgeList = (uint *) malloc(overloadSize * sizeof(uint));
    if (overloadEdgeList == NULL) {
        cout << "overloadEdgeList is null" << endl;
        return;
    }
    gpuErrorcheck(cudaMalloc(&overloadEdgeListD, partOverloadSize * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&degreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&outDegreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&isActiveD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&isStaticActive, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&isOverloadActive, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&valueD, testNumNodes * sizeof(float)));
    gpuErrorcheck(cudaMalloc(&sumD, testNumNodes * sizeof(float)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&overloadLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeListD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadNodePointersD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadDegreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(degreeD, degree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(outDegreeD, outDegree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemset(sumD, 0.0f, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(isActiveD, label, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemset(isStaticActive, 0, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemset(isOverloadActive, 0, testNumNodes * sizeof(uint)));

    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    //setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(isActiveD);

    thrust::device_ptr<unsigned int> ptr_labeling_static(isStaticActive);
    thrust::device_ptr<unsigned int> ptr_labeling_overload(isOverloadActive);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    thrust::device_ptr<unsigned int> ptrOverloadDegree(activeOverloadDegreeD);
    thrust::device_ptr<unsigned int> ptrOverloadPrefixsum(overloadLabelingPrefixD);

    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    uint nodeSum = activeNodesNum;
    ulong overloadEdgeSum = 0;
    auto startCpu = std::chrono::steady_clock::now();
    auto endReadCpu = std::chrono::steady_clock::now();
    long durationReadCpu = 0;

    auto startSwap = std::chrono::steady_clock::now();
    auto endSwap = std::chrono::steady_clock::now();
    long durationSwap = 0;

    auto startGpuProcessing = std::chrono::steady_clock::now();
    auto endGpuProcessing = std::chrono::steady_clock::now();
    long durationGpuProcessing = 0;

    auto startOverloadGpuProcessing = std::chrono::steady_clock::now();
    auto endOverloadGpuProcessing = std::chrono::steady_clock::now();
    long durationOverloadGpuProcessing = 0;

    auto startPreGpuProcessing = std::chrono::steady_clock::now();
    auto endPreGpuProcessing = std::chrono::steady_clock::now();
    long durationPreGpuProcessing = 0;
    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;
    cudaStream_t steamStatic, streamDynamic;
    cudaStreamCreate(&steamStatic);
    cudaStreamCreate(&streamDynamic);
    auto startDynamicMemoryCpy = std::chrono::steady_clock::now();
    auto endDynamicMemoryCpy = std::chrono::steady_clock::now();
    long durationDynamicMemoryCpy = 0;
    //uint cursorStartSwap = staticFragmentNum + 1;
    uint swapValidNodeSum = 0;
    uint swapValidEdgeSum = 0;
    uint swapNotValidNodeSum = 0;
    uint swapNotValidEdgeSum = 0;
    uint visitEdgeSum = 0;
    uint swapInEdgeSum = 0;
    uint partOverloadSum = 0;
    /*uint coutTemp = 0;
    for (uint i = 0; i < testNumNodes; i ++) {
        coutTemp += label[i];
    }
    cout << "coutTemp " << coutTemp << endl;*/
    long TIME = 0;
    int testTimes = 1;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {

        for (uint i = 0; i < testNumNodes; i++) {
            label[i] = 1;
            value[i] = 1.0;
        }
        cudaMemcpy(isInStaticD, isInStatic, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
        gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemset(sumD, 0.0f, testNumNodes * sizeof(uint)));
        gpuErrorcheck(cudaMemcpy(isActiveD, label, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemset(isStaticActive, 0, testNumNodes * sizeof(uint)));
        gpuErrorcheck(cudaMemset(isOverloadActive, 0, testNumNodes * sizeof(uint)));
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        iter = 0;
        auto startProcessing = std::chrono::steady_clock::now();
        while (activeNodesNum > 2) {
            startPreGpuProcessing = std::chrono::steady_clock::now();
            iter++;
            //cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;
            setStaticAndOverloadLabel<<<grid, block>>>(testNumNodes, isActiveD, isStaticActive, isOverloadActive,
                                                       isInStaticD);
            uint staticNodeNum = thrust::reduce(ptr_labeling_static, ptr_labeling_static + testNumNodes);
            if (staticNodeNum > 0) {
                //cout << "iter " << iter << " staticNodeNum " << staticNodeNum << endl;
                thrust::exclusive_scan(ptr_labeling_static, ptr_labeling_static + testNumNodes, ptr_labeling_prefixsum);
                setStaticActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeListD, isStaticActive,
                                                          activeNodeLabelingPrefixD);
            }

            uint overloadNodeNum = thrust::reduce(ptr_labeling_overload, ptr_labeling_overload + testNumNodes);
            uint overloadEdgeNum = 0;
            if (overloadNodeNum > 0) {
                //cout << "iter " << iter << " overloadNodeNum " << overloadNodeNum << endl;
                thrust::exclusive_scan(ptr_labeling_overload, ptr_labeling_overload + testNumNodes,
                                       ptrOverloadPrefixsum);
                setOverloadNodePointerSwap<<<grid, block>>>(testNumNodes, overloadNodeListD, activeOverloadDegreeD,
                                                            isOverloadActive,
                                                            overloadLabelingPrefixD, degreeD);
                thrust::exclusive_scan(ptrOverloadDegree, ptrOverloadDegree + overloadNodeNum,
                                       activeOverloadNodePointersD);

                overloadEdgeNum = thrust::reduce(thrust::device, ptrOverloadDegree,
                                                 ptrOverloadDegree + overloadNodeNum, 0);
                //cout << "iter " << iter << " overloadEdgeNum " << overloadEdgeNum << endl;
                overloadEdgeSum += overloadEdgeNum;
                if (overloadEdgeNum > edgeIterationMax) {
                    edgeIterationMax = overloadEdgeNum;
                }
            }

            endPreGpuProcessing = std::chrono::steady_clock::now();
            durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endPreGpuProcessing - startPreGpuProcessing).count();
            startGpuProcessing = std::chrono::steady_clock::now();
            prSumKernel_static<<<grid, block, 0, steamStatic>>>(staticNodeNum, activeNodeListD,
                                                                staticNodePointerD,
                                                                staticEdgeListD, degreeD, outDegreeD, valueD, sumD);
            if (overloadNodeNum > 0) {
                startCpu = std::chrono::steady_clock::now();
                cudaMemcpyAsync(overloadNodeList, overloadNodeListD, overloadNodeNum * sizeof(uint),
                                cudaMemcpyDeviceToHost,
                                streamDynamic);
                cudaMemcpyAsync(activeOverloadNodePointers, activeOverloadNodePointersD, overloadNodeNum * sizeof(uint),
                                cudaMemcpyDeviceToHost, streamDynamic);
                int threadNum = 20;
                if (overloadNodeNum < 50) {
                    threadNum = 1;
                }
                thread runThreads[threadNum];

                for (int i = 0; i < threadNum; i++) {
                    runThreads[i] = thread(fillDynamic,
                                           i,
                                           threadNum,
                                           0,
                                           overloadNodeNum,
                                           degree,
                                           activeOverloadNodePointers,
                                           nodePointersI,
                                           overloadNodeList,
                                           overloadEdgeList,
                                           edgeList);
                }

                for (unsigned int t = 0; t < threadNum; t++) {
                    runThreads[t].join();
                }
                caculatePartInfoForEdgeList(activeOverloadNodePointers, overloadNodeList, degree, partEdgeListInfoArr,
                                            overloadNodeNum, partOverloadSize, overloadEdgeNum);

                endReadCpu = std::chrono::steady_clock::now();
                durationReadCpu += std::chrono::duration_cast<std::chrono::milliseconds>(endReadCpu - startCpu).count();
                cudaDeviceSynchronize();

                //gpuErrorcheck(cudaPeekAtLastError())
                endGpuProcessing = std::chrono::steady_clock::now();
                durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                        endGpuProcessing - startGpuProcessing).count();
                partOverloadSum += partEdgeListInfoArr.size();
                for (int i = 0; i < partEdgeListInfoArr.size(); i++) {
                    startDynamicMemoryCpy = std::chrono::steady_clock::now();
                    gpuErrorcheck(cudaMemcpy(overloadEdgeListD, overloadEdgeList +
                                                                activeOverloadNodePointers[partEdgeListInfoArr[i].partStartIndex],
                                             partEdgeListInfoArr[i].partEdgeNums * sizeof(uint),
                                             cudaMemcpyHostToDevice))
                    transferSum += partEdgeListInfoArr[i].partEdgeNums;
                    endDynamicMemoryCpy = std::chrono::steady_clock::now();
                    durationDynamicMemoryCpy += std::chrono::duration_cast<std::chrono::milliseconds>(
                            endDynamicMemoryCpy - startDynamicMemoryCpy).count();
                    startOverloadGpuProcessing = std::chrono::steady_clock::now();
                    prSumKernel_dynamic<<<grid, block, 0, streamDynamic>>>(partEdgeListInfoArr[i].partStartIndex,
                                                                           partEdgeListInfoArr[i].partActiveNodeNums,
                                                                           overloadNodeListD,
                                                                           activeOverloadNodePointersD,
                                                                           overloadEdgeListD, degreeD, outDegreeD,
                                                                           valueD,
                                                                           sumD);
                    cudaDeviceSynchronize();
                    gpuErrorcheck(cudaPeekAtLastError())
                    endOverloadGpuProcessing = std::chrono::steady_clock::now();
                    durationOverloadGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                            endOverloadGpuProcessing - startOverloadGpuProcessing).count();
                }
                //gpuErrorcheck(cudaPeekAtLastError())
            } else {
                cudaDeviceSynchronize();
                endGpuProcessing = std::chrono::steady_clock::now();
                durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                        endGpuProcessing - startGpuProcessing).count();
            }
            startPreGpuProcessing = std::chrono::steady_clock::now();
            prKernel_Opt<<<grid, block>>>(testNumNodes, valueD, sumD, isActiveD);
            cudaDeviceSynchronize();
            activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
            nodeSum += activeNodesNum;
            endPreGpuProcessing = std::chrono::steady_clock::now();
            durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endPreGpuProcessing - startPreGpuProcessing).count();
        }
        auto endRead = std::chrono::steady_clock::now();
        durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
        cudaMemcpy(value, valueD, testNumNodes * sizeof(uint), cudaMemcpyDeviceToHost);
        for (int i = 1; i < 31; i++) {
            cout << i << " : " << value[i] << endl;
        }
        transferSum += max_partition_size * sizeof(uint);
        cout << "iterationSum " << iter << endl;
        double edgeIterationAvg = (double) overloadEdgeSum / (double) testNumEdge / iter;
        double edgeIterationMaxAvg = (double) edgeIterationMax / (double) testNumEdge;
        cout << "edgeIterationAvg " << edgeIterationAvg << " edgeIterationMaxAvg " << edgeIterationMaxAvg << endl;
        cout << "transferSum : " << 4 * transferSum << " byte" << endl;
        cout << "finish time : " << durationRead << " ms" << endl;
        cout << "total time : " << testDuration + durationRead << " ms" << endl;
        cout << "cpu time : " << durationReadCpu << " ms" << endl;
        cout << "fact processing time : " << durationGpuProcessing << " ms" << endl;
        cout << "durationOverloadGpuProcessing time : " << durationOverloadGpuProcessing << " ms" << endl;
        cout << "durationDynamicMemoryCpy time : " << durationDynamicMemoryCpy << " ms" << endl;

        cout << "gpu pre processing time : " << durationPreGpuProcessing << " ms" << endl;
        cout << "overloadEdgeSum : " << overloadEdgeSum << " " << endl;
        cout << "partOverloadSum : " << partOverloadSum << " " << endl;
        cout << "nodeSum: " << nodeSum << endl;
        TIME += durationRead;
    }
    cout << "TIME " << (float) TIME / (float) testTimes << endl;
    cudaFree(staticEdgeListD);
    //cudaFree(edgeListOverloadManage);
    cudaFree(degreeD);
    cudaFree(isActiveD);
    cudaFree(valueD);
    cudaFree(activeNodeListD);
    cudaFree(activeNodeLabelingPrefixD);
    cudaFree(activeOverloadNodePointersD);
    cudaFree(activeOverloadDegreeD);
    cudaFree(isInStaticD);
    cudaFree(staticNodePointerD);
    cudaFree(overloadNodeListD);
    cudaFree(overloadEdgeListD);
    cudaFree(isStaticActive);
    cudaFree(isOverloadActive);
    cudaFree(overloadLabelingPrefixD);
    delete[]            label;
    delete[]            degree;
    delete[]            value;
    delete[]            activeNodeList;
    delete[]            activeOverloadNodePointers;
    delete[] isInStatic;
    delete[] overloadNodeList;
    delete[] staticNodePointer;
    delete[] overloadEdgeList;
    partEdgeListInfoArr.clear();
};

/*void prOptDynamicChange(string prPath, float adviseK) {
    cout << "===============prOptDynamicChange==============" << endl;
    ulong edgeIterationSum = 0;
    ulong edgeIterationMax = 0;
    ulong transferSum = 0;
    auto startReadGraph = std::chrono::steady_clock::now();
    GraphMeta<uint> graphInfo{};
    graphInfo.readDataFromFile(std::move(prPath), true);
    unsigned long max_partition_size;
    unsigned long total_gpu_size;

    auto start = std::chrono::steady_clock::now();
    auto startPreCaculate = std::chrono::steady_clock::now();
    //CPU
    long durationRead;
    uint maxStaticNode = 0;
    uint *degree;
    float *value;
    uint *label;
    bool *isInStatic;
    uint *overloadNodeList;
    uint *staticNodePointer;
    uint *staticActiveNodeList;
    uint *activeOverloadNodePointers;
    vector<PartEdgeListInfo> partEdgeListInfoArr;
    uint *overloadEdgeList;
    bool isFromTail = false;
    //GPU
    uint *staticEdgeListD;
    uint *overloadEdgeListD;
    bool *isInStaticD;
    uint *overloadNodeListD;
    uint *staticNodePointerD;
    uint *degreeD;
    uint *outDegreeD;
    // async need two labels
    uint *isActiveD;
    uint *isStaticActive;
    uint *isOverloadActive;
    float *valueD;
    float *sumD;
    uint *activeNodeListD;
    uint *activeNodeLabelingPrefixD;
    uint *overloadLabelingPrefixD;
    uint *activeOverloadNodePointersD;
    uint *activeOverloadDegreeD;

    degree = new uint[testNumNodes];
    value = new float[testNumNodes];
    label = new uint[testNumNodes];
    isInStatic = new bool[testNumNodes];
    overloadNodeList = new uint[testNumNodes];
    staticNodePointer = new uint[testNumNodes];
    staticActiveNodeList = new uint[testNumNodes];
    activeOverloadNodePointers = new uint[testNumNodes];

    getMaxPartitionSize(max_partition_size, total_gpu_size, testNumNodes, adviseK, sizeof(uint), testNumEdge, 15);
    for (uint i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];
    memcpy(staticNodePointer, nodePointersI, testNumNodes * sizeof(uint));

    gpuErrorcheck(cudaMalloc(&staticEdgeListD, max_partition_size * sizeof(uint)));
    auto startmove = std::chrono::steady_clock::now();
    gpuErrorcheck(
            cudaMemcpy(staticEdgeListD, edgeList, max_partition_size * sizeof(uint), cudaMemcpyHostToDevice));
    auto endMove = std::chrono::steady_clock::now();
    long testDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endMove - startmove).count();
    cout << "move duration " << testDuration << endl;
    gpuErrorcheck(cudaMalloc(&isInStaticD, testNumNodes * sizeof(bool)))
    gpuErrorcheck(cudaMalloc(&overloadNodeListD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&staticNodePointerD, testNumNodes * sizeof(uint)))
    gpuErrorcheck(cudaMemcpy(staticNodePointerD, nodePointersI, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));

    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = 1;
        value[i] = 1.0;

        if (nodePointersI[i] < max_partition_size && (nodePointersI[i] + degree[i] - 1) < max_partition_size) {
            isInStatic[i] = true;
            if (i > maxStaticNode) maxStaticNode = i;
        } else {
            isInStatic[i] = false;
        }
    }
    cudaMemcpy(isInStaticD, isInStatic, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
    uint partOverloadSize = total_gpu_size - max_partition_size;
    uint overloadSize = maxStaticNode > 0 ? testNumEdge - nodePointersI[maxStaticNode + 1] : testNumEdge;
    cout << "max_partition_size: " << max_partition_size << "  maxStaticNode: " << maxStaticNode << endl;
    //uint partOverloadSize = max_partition_size / 2;
    cout << " partOverloadSize " << partOverloadSize << " overloadSize " << overloadSize << endl;
    overloadEdgeList = (uint *) malloc(overloadSize * sizeof(uint));
    if (overloadEdgeList == nullptr) {
        cout << "overloadEdgeList is null" << endl;
        return;
    }
    gpuErrorcheck(cudaMalloc(&overloadEdgeListD, partOverloadSize * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&degreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&outDegreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&isActiveD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&isStaticActive, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&isOverloadActive, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&valueD, testNumNodes * sizeof(float)));
    gpuErrorcheck(cudaMalloc(&sumD, testNumNodes * sizeof(float)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&overloadLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeListD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadNodePointersD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadDegreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(degreeD, degree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(outDegreeD, outDegree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemset(sumD, 0.0f, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(isActiveD, label, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemset(isStaticActive, 0, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemset(isOverloadActive, 0, testNumNodes * sizeof(uint)));

    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    //setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(isActiveD);
    thrust::device_ptr<unsigned int> ptr_labeling_static(isStaticActive);
    thrust::device_ptr<unsigned int> ptr_labeling_overload(isOverloadActive);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    thrust::device_ptr<unsigned int> ptrOverloadDegree(activeOverloadDegreeD);
    thrust::device_ptr<unsigned int> ptrOverloadPrefixsum(overloadLabelingPrefixD);

    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    uint nodeSum = activeNodesNum;
    ulong overloadEdgeSum = 0;
    auto startCpu = std::chrono::steady_clock::now();
    auto endReadCpu = std::chrono::steady_clock::now();
    long durationReadCpu = 0;

    auto startSwap = std::chrono::steady_clock::now();
    auto endSwap = std::chrono::steady_clock::now();
    long durationSwap = 0;

    auto startGpuProcessing = std::chrono::steady_clock::now();
    auto endGpuProcessing = std::chrono::steady_clock::now();
    long durationGpuProcessing = 0;

    auto startOverloadGpuProcessing = std::chrono::steady_clock::now();
    auto endOverloadGpuProcessing = std::chrono::steady_clock::now();
    long durationOverloadGpuProcessing = 0;

    auto startDynamicMemoryCpy = std::chrono::steady_clock::now();
    auto endDynamicMemoryCpy = std::chrono::steady_clock::now();
    long durationDynamicMemoryCpy = 0;

    auto startDynamic = std::chrono::steady_clock::now();
    auto endDynamic = std::chrono::steady_clock::now();
    long durationDynamic = 0;

    auto startPreGpuProcessing = std::chrono::steady_clock::now();
    auto endPreGpuProcessing = std::chrono::steady_clock::now();
    long durationPreGpuProcessing = 0;
    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    //cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;
    cudaStream_t steamStatic, streamDynamic;
    cudaStreamCreate(&steamStatic);
    cudaStreamCreate(&streamDynamic);

    //uint cursorStartSwap = staticFragmentNum + 1;
    uint swapValidNodeSum = 0;
    uint swapValidEdgeSum = 0;
    uint swapNotValidNodeSum = 0;
    uint swapNotValidEdgeSum = 0;
    uint visitEdgeSum = 0;
    uint swapInEdgeSum = 0;
    uint partOverloadSum = 0;
    *//*uint coutTemp = 0;
    for (uint i = 0; i < testNumNodes; i ++) {
        coutTemp += label[i];
    }
    cout << "coutTemp " << coutTemp << endl;*//*

    long TIME = 0;
    for (int testIndex = 0; testIndex < 1; testIndex++) {

        for (uint i = 0; i < testNumNodes; i++) {
            label[i] = 1;
            value[i] = 1.0;

            if (nodePointersI[i] < max_partition_size && (nodePointersI[i] + degree[i] - 1) < max_partition_size) {
                isInStatic[i] = true;
                if (i > maxStaticNode) maxStaticNode = i;
            } else {
                isInStatic[i] = false;
            }
        }
        cudaMemcpy(isInStaticD, isInStatic, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
        gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemset(sumD, 0.0f, testNumNodes * sizeof(uint)));
        gpuErrorcheck(cudaMemcpy(isActiveD, label, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemset(isStaticActive, 0, testNumNodes * sizeof(uint)));
        gpuErrorcheck(cudaMemset(isOverloadActive, 0, testNumNodes * sizeof(uint)));
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);

        auto startProcessing = std::chrono::steady_clock::now();
        while (activeNodesNum > 2) {
            startPreGpuProcessing = std::chrono::steady_clock::now();
            iter++;
            *//*if (iter == 2) {
                break;
            }*//*
            //cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;
            setStaticAndOverloadLabel<<<grid, block>>>(testNumNodes, isActiveD, isStaticActive, isOverloadActive,
                                                       isInStaticD);
            uint staticNodeNum = thrust::reduce(ptr_labeling_static, ptr_labeling_static + testNumNodes);
            if (staticNodeNum > 0) {
                //cout << "iter " << iter << " staticNodeNum " << staticNodeNum << endl;
                thrust::exclusive_scan(ptr_labeling_static, ptr_labeling_static + testNumNodes, ptr_labeling_prefixsum);
                setStaticActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeListD, isStaticActive,
                                                          activeNodeLabelingPrefixD);
            }

            uint overloadNodeNum = thrust::reduce(ptr_labeling_overload, ptr_labeling_overload + testNumNodes);
            uint overloadEdgeNum = 0;
            if (overloadNodeNum > 0) {
                //cout << "iter " << iter << " overloadNodeNum " << overloadNodeNum << endl;
                thrust::exclusive_scan(ptr_labeling_overload, ptr_labeling_overload + testNumNodes,
                                       ptrOverloadPrefixsum);
                setOverloadNodePointerSwap<<<grid, block>>>(testNumNodes, overloadNodeListD, activeOverloadDegreeD,
                                                            isOverloadActive,
                                                            overloadLabelingPrefixD, degreeD);
                thrust::exclusive_scan(ptrOverloadDegree, ptrOverloadDegree + overloadNodeNum,
                                       activeOverloadNodePointersD);
                overloadEdgeNum = thrust::reduce(thrust::device, ptrOverloadDegree,
                                                 ptrOverloadDegree + overloadNodeNum, 0);
                //cout << "iter " << iter << " overloadEdgeNum " << overloadEdgeNum << endl;
                long tempSum = (long) overloadEdgeNum * 4;
                cout << "iter " << iter << " overloadEdgeNum " << tempSum << endl;
                overloadEdgeSum += overloadEdgeNum;
                if (overloadEdgeNum > edgeIterationMax) {
                    edgeIterationMax = overloadEdgeNum;
                }
            }
            endPreGpuProcessing = std::chrono::steady_clock::now();
            durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endPreGpuProcessing - startPreGpuProcessing).count();
            startGpuProcessing = std::chrono::steady_clock::now();
            prSumKernel_static<<<grid, block, 0, steamStatic>>>(staticNodeNum, activeNodeListD,
                                                                staticNodePointerD,
                                                                staticEdgeListD, degreeD, outDegreeD, valueD, sumD);
            //cudaDeviceSynchronize();
            if (overloadNodeNum > 0) {
                startCpu = std::chrono::steady_clock::now();
                cudaMemcpyAsync(overloadNodeList, overloadNodeListD, overloadNodeNum * sizeof(uint),
                                cudaMemcpyDeviceToHost,
                                streamDynamic);
                cudaMemcpyAsync(activeOverloadNodePointers, activeOverloadNodePointersD, overloadNodeNum * sizeof(uint),
                                cudaMemcpyDeviceToHost, streamDynamic);
                int threadNum = 20;
                if (overloadNodeNum < 50) {
                    threadNum = 1;
                }
                thread runThreads[threadNum];

                for (int i = 0; i < threadNum; i++) {
                    runThreads[i] = thread(fillDynamic,
                                           i,
                                           threadNum,
                                           0,
                                           overloadNodeNum,
                                           degree,
                                           activeOverloadNodePointers,
                                           nodePointersI,
                                           overloadNodeList,
                                           overloadEdgeList,
                                           edgeList);
                }

                for (unsigned int t = 0; t < threadNum; t++) {
                    runThreads[t].join();
                }
                caculatePartInfoForEdgeList(activeOverloadNodePointers, overloadNodeList, degree, partEdgeListInfoArr,
                                            overloadNodeNum, partOverloadSize, overloadEdgeNum);

                endReadCpu = std::chrono::steady_clock::now();
                durationReadCpu += std::chrono::duration_cast<std::chrono::milliseconds>(endReadCpu - startCpu).count();
                cudaDeviceSynchronize();
                //gpuErrorcheck(cudaPeekAtLastError())
                endGpuProcessing = std::chrono::steady_clock::now();
                durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                        endGpuProcessing - startGpuProcessing).count();
                partOverloadSum += partEdgeListInfoArr.size();
                for (uint i = 0; i < partEdgeListInfoArr.size(); i++) {
                    startDynamic = std::chrono::steady_clock::now();
                    startDynamicMemoryCpy = std::chrono::steady_clock::now();
                    gpuErrorcheck(cudaMemcpy(overloadEdgeListD, overloadEdgeList +
                                                                activeOverloadNodePointers[partEdgeListInfoArr[i].partStartIndex],
                                             partEdgeListInfoArr[i].partEdgeNums * sizeof(uint),
                                             cudaMemcpyHostToDevice))
                    transferSum += partEdgeListInfoArr[i].partEdgeNums;
                    cudaDeviceSynchronize();
                    endDynamicMemoryCpy = std::chrono::steady_clock::now();
                    durationDynamicMemoryCpy += std::chrono::duration_cast<std::chrono::milliseconds>(
                            endDynamicMemoryCpy - startDynamicMemoryCpy).count();
                    startOverloadGpuProcessing = std::chrono::steady_clock::now();
                    prSumKernel_dynamic<<<grid, block, 0, streamDynamic>>>(partEdgeListInfoArr[i].partStartIndex,
                                                                           partEdgeListInfoArr[i].partActiveNodeNums,
                                                                           overloadNodeListD,
                                                                           activeOverloadNodePointersD,
                                                                           overloadEdgeListD, degreeD, outDegreeD,
                                                                           valueD,
                                                                           sumD);
                    cudaDeviceSynchronize();
                    gpuErrorcheck(cudaPeekAtLastError())
                    endOverloadGpuProcessing = std::chrono::steady_clock::now();
                    durationOverloadGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                            endOverloadGpuProcessing - startOverloadGpuProcessing).count();
                    endDynamic = std::chrono::steady_clock::now();
                    durationDynamic += std::chrono::duration_cast<std::chrono::milliseconds>(
                            endDynamic - startDynamic).count();
                }
                //gpuErrorcheck(cudaPeekAtLastError())
            } else {
                cudaDeviceSynchronize();
                endGpuProcessing = std::chrono::steady_clock::now();
                durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                        endGpuProcessing - startGpuProcessing).count();
            }
            startPreGpuProcessing = std::chrono::steady_clock::now();
            prKernel_Opt<<<grid, block>>>(testNumNodes, valueD, sumD, isActiveD);
            cudaDeviceSynchronize();
            activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
            nodeSum += activeNodesNum;
            endPreGpuProcessing = std::chrono::steady_clock::now();
            durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                    endPreGpuProcessing - startPreGpuProcessing).count();
        }
        auto endRead = std::chrono::steady_clock::now();
        durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
        *//*cudaMemcpy(value, valueD, testNumNodes * sizeof(uint), cudaMemcpyDeviceToHost);
        for (int i = 1; i < 31; i++) {
            cout << i << " : " << value[i] << endl;
        }*//*
        transferSum += max_partition_size * sizeof(uint);
        cout << "iterationSum " << iter << endl;
        double edgeIterationAvg = (double) overloadEdgeSum / (double) testNumEdge / iter;
        double edgeIterationMaxAvg = (double) edgeIterationMax / (double) testNumEdge;
        cout << "edgeIterationAvg " << edgeIterationAvg << " edgeIterationMaxAvg " << edgeIterationMaxAvg << endl;
        cout << "transferSum : " << 4 * transferSum << " byte" << endl;
        cout << "finish time : " << durationRead << " ms" << endl;
        cout << "total time : " << testDuration + durationRead << " ms" << endl;
        cout << "cpu time : " << durationReadCpu << " ms" << endl;
        cout << "fact processing time : " << durationGpuProcessing << " ms" << endl;
        cout << "durationOverloadGpuProcessing time : " << durationOverloadGpuProcessing << " ms" << endl;
        cout << "durationDynamicMemoryCpy time : " << durationDynamicMemoryCpy << " ms" << endl;
        cout << "durationDynamictime : " << durationDynamic << " ms" << endl;

        cout << "gpu pre processing time : " << durationPreGpuProcessing << " ms" << endl;
        //cout << "overloadEdgeSum : " << overloadEdgeSum << " " << endl;
        cout << "partOverloadSum : " << partOverloadSum << " " << endl;
        //cout << "nodeSum: " << nodeSum << endl;
        TIME += durationRead;
    }
    cout << "TIME " << (float) TIME / (float) 10;
    cudaFree(staticEdgeListD);
    //cudaFree(edgeListOverloadManage);
    cudaFree(degreeD);
    cudaFree(isActiveD);
    cudaFree(valueD);
    cudaFree(activeNodeListD);
    cudaFree(activeNodeLabelingPrefixD);
    cudaFree(activeOverloadNodePointersD);
    cudaFree(activeOverloadDegreeD);
    cudaFree(isInStaticD);
    cudaFree(staticNodePointerD);
    cudaFree(overloadNodeListD);
    cudaFree(overloadEdgeListD);
    cudaFree(isStaticActive);
    cudaFree(isOverloadActive);
    cudaFree(overloadLabelingPrefixD);
    delete[]            label;
    delete[]            degree;
    delete[]            value;
    delete[]            staticActiveNodeList;
    delete[]            activeOverloadNodePointers;
    delete[] isInStatic;
    delete[] overloadNodeList;
    delete[] staticNodePointer;
    delete[] overloadEdgeList;
    partEdgeListInfoArr.clear();
};*/

