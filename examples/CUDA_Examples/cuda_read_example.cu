/*
 * cuda_read_example.cu
 *
 * Example of reading a genome file using the
 * OO_Parallel_IO CUDA / GPUDirect Storage backend.
 *
 * Build:
 * nvcc -std=c++17 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcufile -lcudart -o cuda_read_example cuda_read_example.cu
 *
 * Run:
 * ./cuda_read_example
 *
 * @author: Josh Eilu
 * @date: Summer 2026
 */

#include "../../OO_IO/include/CUDAIO.h"

int main() {

    const std::string inputFile = "../../data/Felis_catus_9.dna.plain";

    // Open the genome file for reading

    gpuReader<char> reader;

    reader.open(inputFile, O_RDONLY | O_DIRECT);

    // Allocate GPU buffer for reading
    reader.setDeviceBufferSize(1024 * 1024 * 1024 * 6ULL); // 6 GB buffer

    const std::size_t numChunks = reader.getNumChunks();
    for (int i = 0; i < numChunks; ++i) {
        char *gpuData = reader.readChunk(i);

        
    }

    // Process gpuData here

    // gpuData points to GPU memory containing the genome.
    //
    // A CUDA kernel could be launched here:
    //
    // processGenome<<<blocks, threads>>>(
    //     gpuData,
    //     reader.getNumItemsInFile()
    // );

    // Close the reader

    reader.close();

    return 0;
}