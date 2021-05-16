#include "test.cuh"

typedef double real_t;

#include <chrono>

struct n_t {
    n_t ( int idx, float v ) : idx ( idx ), v ( v ) { }
    bool operator < ( n_t const & o ) const { return v < o.v; }
    int idx;
    float v;
};

void initRandomMatrix(real_t * matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = (double)rand();
    }
}

int main(int argc, char **argv)
{
    testShare<<<1,1>>>();
    cudaDeviceSynchronize();

    testRunThread testRunThread;
    testRunThread.run();
    testRunThread.sync();

    printf("[DTNN CUBLAS] - Starting...\n");
    int devID = 0;
    cudaDeviceProp deviceProp;

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d max_Ss %d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);
    int threshold = 100;
    int block_size = 32;
    unsigned int matrixWidth = 4 * block_size;
    unsigned int matrixHeight = 4 * block_size;
    unsigned int matrixSize = matrixWidth * matrixHeight;

    priority_queue < n_t > * pq = new priority_queue < n_t >[matrixWidth];
    real_t* deviceA;
    real_t* hostA;
    real_t* deviceC;
    real_t* hostC;
    real_t* pinnedC;
    real_t *hostNorm;
    real_t *deviceNorm;
    hostA = (real_t*) malloc(matrixSize * sizeof(real_t));
    hostC = (real_t*) malloc(matrixSize * sizeof(real_t));
    hostNorm = (real_t*) malloc(matrixWidth * sizeof(real_t));
    initRandomMatrix(hostA, matrixSize);
    checkCudaErrors(cudaMalloc((void **) &deviceA, matrixSize * sizeof(real_t)));
    checkCudaErrors(cudaMalloc((void **) &deviceC, matrixSize * sizeof(real_t)));
    checkCudaErrors(cudaHostAlloc((void **) &pinnedC, matrixSize * sizeof(real_t), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc((void **) &deviceNorm, matrixWidth * sizeof(real_t)));
    cudaMemcpy(deviceA, hostA, matrixSize * sizeof(real_t), cudaMemcpyHostToDevice);
    printf("Computing result using CUBLAS...");
    const real_t alpha = (real_t) -2.0;
    const real_t beta = (real_t) 1.0;
    cublasHandle_t handle;
    cudaEvent_t start, stop;

    checkCudaErrors(cublasCreate(&handle));
    // Allocate CUDA events that we'll use for timing
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    for ( int i = 0; i < matrixWidth; ++i ) {
        checkCudaErrors(cublasDnrm2(handle, matrixWidth, deviceA + i
                , matrixWidth, hostNorm + i));
    }

    printf("Test common device memory.\n");
    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    for ( int i = 0; i < matrixWidth; ++i ) {
        for( int j = 0; j <= matrixWidth; ++j ) {
            hostC[ i * matrixWidth + j] = hostNorm[i] + hostNorm[j];
        }
    }

    cudaMemcpy(deviceC, hostC, matrixSize * sizeof(real_t), cudaMemcpyHostToDevice);
    long durationDevice = 0;
    long durationDeviceCPU = 0;
    for ( int r = 0; r < 10; ++r ) {
        auto startRead = std::chrono::steady_clock::now();
        checkCudaErrors ( cublasDsyrk (
                handle,
                CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                matrixWidth, matrixWidth,
                &alpha,
                deviceA, matrixWidth,
                &beta,
                deviceC, matrixWidth
        ) );

        checkCudaErrors ( cudaDeviceSynchronize ( ) );
        auto endRead = std::chrono::steady_clock::now();
        long durationRead = std::chrono::duration_cast<std::chrono::microseconds>(endRead - startRead).count();
        durationDevice += durationRead;
        startRead = std::chrono::steady_clock::now();
        cudaMemcpy(hostC, deviceC, matrixSize * sizeof(real_t)
                , cudaMemcpyDeviceToHost);
        for ( int j = 0; j < matrixWidth; ++j ) {
            for ( int i = 0; i < j; ++i ) {
                float v = ( float ) hostC[j * matrixWidth + i];
                pq[i].push ( n_t ( j, v ) );
                if ( pq[i].size ( ) > threshold ) { pq[i].pop ( ); }
                pq[j].push ( n_t ( i, v ) );
                if ( pq[j].size ( ) > threshold ) { pq[j].pop ( ); }
            }
        }
        endRead = std::chrono::steady_clock::now();
        durationRead = std::chrono::duration_cast<std::chrono::microseconds>(endRead - startRead).count();
        durationDeviceCPU += durationRead;
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    //printf("Test common device memory done.\n");

    //printf("Time= %.3f milli sec cublas time is %d micro sec cpu time is %d micro seconds\n",msecTotal, durationDevice, durationDeviceCPU);


    printf("Test pinned memory.\n");
    checkCudaErrors(cudaEventRecord(start, NULL));

    for ( int i = 0; i < matrixWidth; ++i ) {
        for( int j = 0; j <= matrixWidth; ++j ) {
            pinnedC[ i * matrixWidth + j] = hostNorm[i] + hostNorm[j];
        }
    }

    long durationPinned = 0;
    long durationPinnedCPU = 0;
    for ( int r = 0; r < 10; ++r ) {
        auto startRead = std::chrono::steady_clock::now();
        checkCudaErrors ( cublasDsyrk (
                handle,
                CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                matrixWidth, matrixWidth,
                &alpha,
                deviceA, matrixWidth,
                &beta,
                pinnedC, matrixWidth
        ) );

        checkCudaErrors ( cudaDeviceSynchronize ( ) );
        auto endRead = std::chrono::steady_clock::now();
        long durationRead = std::chrono::duration_cast<std::chrono::microseconds>(endRead - startRead).count();
        durationPinned += durationRead;
        startRead = std::chrono::steady_clock::now();
        for ( int j = 0; j < matrixWidth; ++j ) {
            for ( int i = 0; i < j; ++i ) {
                float v = ( float ) pinnedC[j * matrixWidth + i];
                pq[i].push ( n_t ( j, v ) );
                if ( pq[i].size ( ) > threshold ) { pq[i].pop ( ); }
                pq[j].push ( n_t ( i, v ) );
                if ( pq[j].size ( ) > threshold ) { pq[j].pop ( ); }
            }
        }
        endRead = std::chrono::steady_clock::now();
        durationRead = std::chrono::duration_cast<std::chrono::microseconds>(endRead - startRead).count();
        durationPinnedCPU += durationRead;
    }


    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    printf("Test pinned memory done.\n");
    printf("Time= %.3f milli seconds, cublas time is %d micro sec cpu time is %d micro seconds\n",msecTotal, durationPinned, durationPinnedCPU);

    printf("Test cagtp.\n");
    checkCudaErrors(cudaEventRecord(start, NULL));
    for ( int i = 0; i < matrixWidth; ++i ) {
        for( int j = 0; j <= matrixWidth; ++j ) {
            pinnedC[ i * matrixWidth + j] = hostNorm[i] + hostNorm[j];
        }
    }

    MatrixData matrixData;
    matrixData.matrixA = deviceA;
    matrixData.matrixC = pinnedC;
    matrixData.matrixWidth = matrixWidth;
    matrixData.alpha = alpha;
    matrixData.beta = beta;
    TaskPoolGpu gtp(1, matrixData, 0);
    gtp.run();
    TaskDetail task;
    long durationGTPCPU = 0;
    long durationGTP = 0;
    for ( int r = 0; r < 10; ++r ) {
        int cnt = 0;
        TaskDetail output;
        for ( int j = 0; j < matrixWidth/BLOCK_SIZE; ++j ) for ( int i = 0; i <= j; ++i ) {
                task.blockX = i;
                task.blockY = j;
                gtp.pushTask(task);
                ++cnt;
            }

        auto startRead = std::chrono::steady_clock::now();
        bool isFirst = true;
        while ( cnt-- ) {
            if (!gtp.popFinish(output)) {
                cnt++;
            } else {
                if(isFirst) {
                    isFirst = false;
                    auto endRead1 = std::chrono::steady_clock::now();
                    long durationRead1 = std::chrono::duration_cast<std::chrono::microseconds>(endRead1 - startRead).count();
                    durationGTP += durationRead1;
                }
                auto startReadCPU = std::chrono::steady_clock::now();
                int jb  = task.blockY * BLOCK_SIZE;
                int je = jb + BLOCK_SIZE;
                int ib  = task.blockX * BLOCK_SIZE;
                int ie = ib + BLOCK_SIZE;
                if ( output.blockY == output.blockX ) {
                    for ( int j = jb; j < je; ++j ) for ( int i = ib; i < j; ++i ) {
                            float v = pinnedC[j * matrixWidth + i];
                            pq[i].push ( n_t ( j, v ) );
                            if ( pq[i].size ( ) > threshold ) { pq[i].pop ( ); }
                            pq[j].push ( n_t ( i, v ) );
                            if ( pq[j].size ( ) > threshold ) { pq[j].pop ( ); }
                        }
                } else {
                    for ( int j = jb; j < je; ++j ) for ( int i = ib; i < ie; ++i ) {
                            float v = pinnedC[j * matrixWidth + i];
                            pq[i].push ( n_t ( j, v ) );
                            if ( pq[i].size ( ) > threshold ) { pq[i].pop ( ); }
                            pq[j].push ( n_t ( i, v ) );
                            if ( pq[j].size ( ) > threshold ) { pq[j].pop ( ); }
                        }
                }
                auto endReadCPU = std::chrono::steady_clock::now();
                long durationReadCPU = std::chrono::duration_cast<std::chrono::microseconds>(endReadCPU - startReadCPU).count();
                durationGTPCPU += durationReadCPU;
            }
        }

    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    gtp.setExit();
    gtp.sync();
    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    printf("Test cagtp done.\n");
    printf("Time= %.3f milli seconds, cagtp time is %d micro seconds  cpu time is %d micro seconds\n", msecTotal, durationGTP, durationGTPCPU);
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    // Destroy the handle
    checkCudaErrors(cublasDestroy(handle));
    return 0;
}
