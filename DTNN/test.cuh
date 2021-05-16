//
// Created by gxl on 2020/11/10.
//

#ifndef DTNN_TEST_CUH
#define DTNN_TEST_CUH

#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <ctime>
#include <queue>
#include <list>
#include <thread>
#include <chrono>
#include <unistd.h>
// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

using namespace std;

class testRunThread {
private:
    int threadNum = 9999;
    thread runThread;
public:
    void run() {
        runThread = thread([&](void *args) {

            cout << " hello world " << endl;
            testRunThread *testArgs = (testRunThread*) args;
            cout << "thread NUm " << threadNum << endl;
            return;
        }, this);

    }

    void sync() {
        runThread.join();
    }
};

__global__ void testShare() {
    int tid = blockIdx.x;
    printf("===========testShare==========\n");
}

const int BLOCK_SIZE = 32;

enum TaskSlotStatus {
    IDLE, FINISHED, GPU_CONCERN, READY, EXIT,
};

//为了编程方便暂时使用方阵
struct MatrixData {
    double *matrixA;
    double *matrixC;
    double alpha;
    double beta;
    int matrixWidth;
};

struct TaskDetail {
    int blockX;
    int blockY;
};

struct TaskSlot {
    TaskDetail taskDetail;
    TaskSlotStatus status;
};

__device__ void doTask(TaskDetail task, MatrixData const &data);
__global__ void kernel(TaskSlot *taskSlot, MatrixData data);

class TaskPoolGpu {
private:
    size_t numBlocks = 1;
    TaskSlot *taskSlots;
    std::queue<TaskDetail> inputQueue;
    std::queue<TaskDetail> outputQueue;
    volatile bool isExit = false;
    MatrixData matrixData;
    int device;
    thread gpuScheduler;
public:
    TaskPoolGpu(size_t blocksOfSm, MatrixData matrixData, int device) :
            matrixData(matrixData), device(device) {
        int devID = device;
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
               deviceProp.name, deviceProp.major, deviceProp.minor);
        numBlocks = blocksOfSm * deviceProp.multiProcessorCount;
        checkCudaErrors(
                cudaHostAlloc((void**) &taskSlots, numBlocks * sizeof(TaskSlot),
                              cudaHostAllocDefault));
    }

    void run() {
        for (int i = 0; i < numBlocks; ++i) {
            TaskSlot *p = (struct TaskSlot*) (taskSlots + i);
            p->status = IDLE;
        }

        gpuScheduler = thread([&](void *args) {
            std::cout << "gpuScheduler" << std::endl;
            TaskPoolGpu *gtp = (TaskPoolGpu*) args;
            for (int i = 0;; i = (i + 1) % gtp->numBlocks) {
                volatile TaskSlot *p = (volatile TaskSlot*) (gtp->taskSlots + i);
                if (FINISHED == p->status) {
                    TaskSlot task_slot = * (TaskSlot*) p;
                    gtp->outputQueue.push(task_slot.taskDetail);
                    p->status = IDLE;
                }
                if (IDLE == p->status) {
                    if (inputQueue.size() > 0) {
                        TaskDetail taskDetail = inputQueue.front();
                        inputQueue.pop();
                        p->taskDetail.blockX = taskDetail.blockX;
                        p->taskDetail.blockY = taskDetail.blockY;
                        p->status = READY;
                    } else if (gtp->isExit) {
                        break;
                    }
                }
            }

            std::list<int> running_blocks;
            for (int i = 0; i < gtp->numBlocks; ++i) {
                running_blocks.push_back(i);
            }

            for (std::list<int>::iterator it = running_blocks.begin();
                 !running_blocks.empty();) {
                if (running_blocks.end() == it)
                    it = running_blocks.begin();
                volatile TaskSlot *p = (volatile TaskSlot*) (gtp->taskSlots
                                                             + *it);
                if (FINISHED == p->status) {
                    TaskSlot task_slot = *(TaskSlot*) p;
                    gtp->outputQueue.push(task_slot.taskDetail);
                    p->status = IDLE;
                }
                if (IDLE == p->status) {
                    p->status = EXIT;
                    running_blocks.erase(it++);
                    continue;
                }
                ++it;
            }
            printf("GTP scheduler exiting...\n");
        }, this);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
        kernel<<<numBlocks, dim_block>>>(taskSlots, matrixData);
    }

    void sync() {
        gpuScheduler.join();
        cudaDeviceSynchronize();
    }

    void setExit() {
        isExit = true;
    }

    void pushTask(TaskDetail task) {
        TaskDetail tempTask;
        tempTask.blockX = task.blockX;
        tempTask.blockY = task.blockY;
        inputQueue.push(tempTask);
    }

    int getInputQueueSize() {
        return inputQueue.size();
    }

    int getOutputQueueSize() {
        return outputQueue.size();
    }

    bool popFinish(TaskDetail &finishTask) {
        if (outputQueue.size() > 0) {
            TaskDetail task = outputQueue.front();
            outputQueue.pop();
            finishTask.blockX = task.blockX;
            finishTask.blockY = task.blockY;
            return true;
        }
        return false;
    }
    ~TaskPoolGpu() {
        cudaFreeHost((void*) taskSlots);
    }
};

__global__ void kernel(TaskSlot *taskSlot, MatrixData data) {
    volatile TaskSlot *taskSlotP = taskSlot + blockIdx.x;

    while (1) {
        if (0 == threadIdx.x && 0 == threadIdx.y && 0 == threadIdx.z) {
            while (taskSlotP->status < GPU_CONCERN) {
            }
        }
        __syncthreads();

        if (EXIT == taskSlotP->status)
            return;
        TaskDetail taskDetail1;
        taskDetail1.blockX = taskSlotP->taskDetail.blockX;
        taskDetail1.blockY = taskSlotP->taskDetail.blockY;
        doTask(taskDetail1, data);
        __syncthreads();

        if (0 == threadIdx.x && 0 == threadIdx.y && 0 == threadIdx.z) {
            taskSlotP->status = FINISHED;
        }
    }
}

__device__ void doTask(TaskDetail task, MatrixData const &data) {
    int bx = task.blockX;
    int by = task.blockY;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = data.matrixWidth * BLOCK_SIZE * by;
    int aEnd = aBegin + data.matrixWidth - 1;
    int aStep = BLOCK_SIZE;

    //因为是矩阵乘其转置，因此B直接可从A矩阵中取
    int bBegin = data.matrixWidth * BLOCK_SIZE * bx;
    int bEnd = bBegin + data.matrixWidth - 1;
    int bStep = BLOCK_SIZE;

    double cSub = 0;
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__
        float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__
        float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = data.matrixA[a + data.matrixWidth * ty + tx];
        Bs[ty][tx] = data.matrixA[b + data.matrixWidth * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            cSub += As[ty][k] * Bs[tx][k];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = data.matrixWidth * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    data.matrixC[c + data.matrixWidth * ty + tx] = data.matrixC[c
                                                                + data.matrixWidth * ty + tx] + -2 * cSub;

}

__global__ void test() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        printf("hello world \n");
    }
}




#endif //DTNN_TEST_CUH
