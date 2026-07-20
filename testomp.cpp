#include <iostream>
#include <span>
#include <string>

#include <omp.h>

#include "OO_IO/include/ThreadsIO.h"

/*
 * Test ThreadReader using OpenMP threads.
 *
 * Usage:
 *   ./threadReaderTest <binary-file>
 */
int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <binary-file>\n";
        return 1;
    }

    const std::string fileName = argv[1];
    const int NUM_THREADS = 5;

    unsigned long long totalItemsRead = 0;
    long double checksum = 0.0L;

    std::cout << "\n====================================\n";
    std::cout << "OPENMP THREADREADER TEST\n";
    std::cout << "====================================\n";
    std::cout << "File: " << fileName << '\n';
    std::cout << "Requested threads: " << NUM_THREADS << "\n\n";

    double startTime = omp_get_wtime();

#pragma omp parallel num_threads(NUM_THREADS) reduction(+ : totalItemsRead, checksum)
    {
        int id = omp_get_thread_num();
        int numThreads = omp_get_num_threads();

        ThreadReader<double> reader(id, numThreads);

        reader.open(fileName);

        std::span<const double> chunk = reader.readChunk();

        // Use the span while the reader mapping is still open.
        for (double value : chunk) {
            checksum += value;
        }

        totalItemsRead += chunk.size();

#pragma omp critical
        {
            std::cout << "Thread " << id
                      << " read " << chunk.size()
                      << " doubles, starting at item "
                      << reader.getFirstItemOffset()
                      << '\n';
        }

        // The span must not be used after this call.
        reader.close();
    }

    double elapsedTime = omp_get_wtime() - startTime;

    std::cout << "\n====================================\n";
    std::cout << "RESULTS\n";
    std::cout << "====================================\n";
    std::cout << "Total doubles read: " << totalItemsRead << '\n';
    std::cout << "Checksum: " << checksum << '\n';
    std::cout << "Elapsed time: " << elapsedTime << " seconds\n";

    return 0;
}