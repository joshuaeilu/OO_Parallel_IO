/*
 * cuda_read_write_example.cu
 *
 * Example of reading and writing a genome file using the
 * OO_Parallel_IO CUDA / GPUDirect Storage backend.
 *
 * The genome is read directly from storage into GPU memory
 * and then written directly from GPU memory to a new file.
 *
 * Build:
 * nvcc -O3 -std=c++17 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcufile -lcudart -o cuda_read_write_example cuda_read_write_example.cu
 *
 * Run:
 * ./cuda_read_write_example
 *
 * @author: Josh Eilu
 * @date: Summer 2026
 */

#include "../../OO_IO/include/CUDAIO.h"


#include <iostream>
#include <string>

int main() {

    const std::string inputFile = "../../data/Felis_catus_9.dna.plain";

    const std::string outputFile = "../../Felis_catus_9.cuda.copy.dna.plain";

    // Read genome directly into GPU memory

    CUDAReader<char> reader;

    reader.open(inputFile, O_RDONLY | O_DIRECT);

    char *gpuData = reader.readToGPU();

    const long fileSize = reader.getFileSize();

    const std::size_t numItems = static_cast<std::size_t>(reader.getNumItemsInFile());

    std::cout << "Read " << numItems << " characters into GPU memory.\n";

    // Process genome on GPU here, if desired

    // For example:
    //
    // processGenome<<<blocks, threads>>>(
    //     gpuData,
    //     numItems
    // );
    //
    // checkResult(cudaGetLastError());
    // checkResult(cudaDeviceSynchronize());

    // Write the same GPU data to a new file

    CUDAWriter<char> writer(fileSize, outputFile);

    writer.writeToFile(gpuData, numItems);

    std::cout << "Wrote " << numItems << " characters from GPU memory "
              << "to the output file.\n";

              // Close writer first

    writer.close();

    // gpuData belongs to the reader, so close the reader
    // only after the writer has finished using gpuData.
    reader.close();

    return 0;
}