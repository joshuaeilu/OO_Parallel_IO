/*
 * thread_write_example.cpp
 *
 * Example of writing genome data using the
 * OO_Parallel_IO Threads backend.
 *
 * Each thread creates a chunk of genome data and writes
 * its assigned portion to a shared output file.
 *
 * Build:
 * g++ -std=c++20 -fopenmp -pthread -o thread_write_example thread_write_example.cpp
 *
 * Run:
 * ./thread_write_example
 *
 * @author: Josh Eilu
 * @date: Summer 2026
 */

#include "../../OO_IO/include/ThreadsIO.h"

#include <fcntl.h>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

int main() {

    const int numThreads = 4;

    const std::string outputFile = "../../thread_genome_output.dna";

    // Each thread will write this genome sequence.
    const std::string genomeSequence = "ACGTACGTACGTACGT";

    // Total output file size in bytes.
    const long fileSize = static_cast<long>(genomeSequence.size() * numThreads);

#pragma omp parallel num_threads(numThreads)
    {
        const int id = omp_get_thread_num();

        const int numPEs = omp_get_num_threads();

        // Create this thread's genome chunk

        std::vector<char> genomeChunk(genomeSequence.begin(), genomeSequence.end());

        // Open the output file for parallel writing

        ThreadWriter<char> writer(id, numPEs, fileSize);

        writer.open(outputFile, O_RDWR);

        // Write this thread's assigned chunk

        writer.writeChunk(genomeChunk);

#pragma omp critical
        {
            std::cout << "Thread " << id << " wrote " << genomeChunk.size()
                      << " characters.\n";
        }

        // Close the writer

        writer.close();
    }

    return 0;
}