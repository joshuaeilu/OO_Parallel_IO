/*
 * cuda_write_example.cu
 *
 * Example of writing genome data directly from GPU memory
 * using the OO_Parallel_IO CUDA / GPUDirect Storage backend.
 *
 * Build:
 * nvcc -O3 -std=c++17 \
 *     -I/usr/local/cuda/include \
 *     -L/usr/local/cuda/lib64 \
 *     -lcufile -lcudart \
 *     -o cuda_write_example cuda_write_example.cu
 *
 * Run:
 * ./cuda_write_example
 *
 * @author: Josh Eilu
 * @date: Summer 2026
 */

#include "../../OO_IO/include/CUDAIO.h"

#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <vector>

int main() {

    const std::string outputFile = "../../cuda_genome_output.dna";

    constexpr std::size_t numItems = 4096;

    // Create genome data on the CPU

    std::vector<char> genomeData(numItems);

    const char bases[] = {'A', 'C', 'G', 'T'};

    for (std::size_t i = 0; i < numItems; ++i) {
        genomeData[i] = bases[i % 4];
    }

    // Allocate GPU memory

    char *gpuData = nullptr;

    checkResult(cudaMalloc(reinterpret_cast<void **>(&gpuData), numItems * sizeof(char)));

    // Copy genome data to GPU

    checkResult(cudaMemcpy(gpuData, genomeData.data(), numItems * sizeof(char),
                           cudaMemcpyHostToDevice));

    // Write GPU memory directly to storage

    CUDAWriter<char> writer(static_cast<long>(numItems * sizeof(char)), outputFile);

    writer.writeToFile(gpuData, numItems);

    std::cout << "Wrote " << numItems << " genome characters directly "
              << "from GPU memory.\n";

    // Close and clean up

    writer.close();

    checkResult(cudaFree(gpuData));

    return 0;
}