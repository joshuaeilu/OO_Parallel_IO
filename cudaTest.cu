#include "OO_IO/include/CUDAIO.h"

#include <cstdio>

__global__
void printData(double* data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 10) {
        printf("data[%d] = %f\n", i, data[i]);
    }
}

int main() {

    CUDAReader<double> reader("./files/1b-doubles.bin");
    double* gpuData = reader.readChunksToGPU();
    printData<<<1, 10>>>(gpuData);
    cudaDeviceSynchronize();
    reader.close();

    return 0;
}