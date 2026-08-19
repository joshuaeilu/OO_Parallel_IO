/*
 * thread_read_write_example.cpp
 *
 * Example of reading and writing a genome file using the
 * OO_Parallel_IO Threads backend.
 *
 * Build:
 * g++ -std=c++20 -fopenmp -pthread -o thread_read_write_example thread_read_write_example.cpp
 *
 * Run:
 * ./thread_read_write_example
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

    const int numThreads = 6;

    const std::string inputFile = "../../data/Felis_catus_9.dna.plain";

    const std::string outputFile = "../../Felis_catus_9.thread.copy.dna.plain";

    double readStart = 0.0;
    double writeStart = 0.0;
    double totalStart = 0.0;

    double readTime = 0.0;
    double writeTime = 0.0;
    double totalTime = 0.0;

#pragma omp parallel num_threads(numThreads)
    {
        const int id = omp_get_thread_num();
        const int numPEs = omp_get_num_threads();

        std::vector<char> genomeChunk;
        long fileSize = 0;

        // -----------------------------------------------------
        // Start total and read timers
        // -----------------------------------------------------

#pragma omp single
        {
            totalStart = omp_get_wtime();
            readStart = omp_get_wtime();
        }

        // -----------------------------------------------------
        // Read this thread's portion of the genome
        // -----------------------------------------------------

        {
            ThreadReader<char> reader(id, numPEs);

            reader.open(inputFile, O_RDONLY);

            genomeChunk = reader.readChunk();
            fileSize = reader.getFileSize();

            reader.close();
        }

#pragma omp barrier

#pragma omp single
        {
            readTime = omp_get_wtime() - readStart;
        }

        // -----------------------------------------------------
        // Start write timer
        // -----------------------------------------------------

#pragma omp single
        {
            writeStart = omp_get_wtime();
        }

        // -----------------------------------------------------
        // Write this thread's portion of the genome
        // -----------------------------------------------------

        {
            ThreadWriter<char> writer(id, numPEs, fileSize);

            writer.open(outputFile, O_RDWR);

            writer.writeChunk(genomeChunk);

            writer.close();
        }

#pragma omp barrier

#pragma omp single
        {
            writeTime = omp_get_wtime() - writeStart;
            totalTime = omp_get_wtime() - totalStart;
        }

        // -----------------------------------------------------
        // Display thread information
        // -----------------------------------------------------

#pragma omp critical
        {
            std::cout << "Thread " << id << " read and wrote "
                      << genomeChunk.size() << " characters.\n";
        }
    }

    // ---------------------------------------------------------
    // Display timing results
    // ---------------------------------------------------------

    std::cout << "\nThreads:    " << numThreads << '\n';
    std::cout << "Read time:  " << readTime << " seconds\n";
    std::cout << "Write time: " << writeTime << " seconds\n";
    std::cout << "Total time: " << totalTime << " seconds\n";

    return 0;
}