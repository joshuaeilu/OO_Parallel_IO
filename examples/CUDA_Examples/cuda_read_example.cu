/*
 * cuda_read_example.cu
 *
 * Example of reading a genome file using the
 * OO_Parallel_IO CUDA / GPUDirect Storage backend.
 *
 * Build:
 * nvcc -std=c++17 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcufile -lcudart -o
 * cuda_read_example cuda_read_example.cu
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

    CUDAReader<char> reader;

    reader.open(inputFile, O_RDONLY | O_DIRECT);

    // Read the genome directly into GPU memory

    char *gpuData = reader.readToGPU();

    std::cout << "Read " << reader.getNumItemsInFile()
              << " characters into GPU memory.\n";

    std::cout << "File size: " << reader.getFileSize() << " bytes.\n";

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