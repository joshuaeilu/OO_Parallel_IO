/*
 * thread_read_example.cpp
 *
 * Example of reading a genome file using the
 * OO_Parallel_IO Threads backend.
 *
 * Build:
 * g++ -std=c++20 -fopenmp -pthread -o thread_read_example thread_read_example.cpp
 *
 * Run:
 * ./thread_read_example
 *
 * @author: Josh Eilu
 * @date: Summer 2026
 */

#include "../../OO_IO/include/ThreadsIO.h"
#include <omp.h>


int main() {

    const int numThreads = 4;

    // Read the genome file in parallel

#pragma omp parallel num_threads(numThreads)
    {
        const int id = omp_get_thread_num();
        const int numPEs = omp_get_num_threads();

        // Create a reader for this thread.
        ThreadReader<char> reader(id, numPEs);

        // Open the genome file for parallel reading.
        reader.open("../../data/Felis_catus_9.dna.plain", O_RDONLY);

        // Read this thread's portion of the genome.
        std::vector<char> chunk = reader.readChunk();

        // Process this thread's chunk here

#pragma omp critical
        { std::cout << "Thread " << id << " read " << chunk.size() << " characters.\n"; }

        // Close this thread's reader.
        reader.close();
    }

    return 0;
}