#include "OO_IO.h"

#include <iostream>
#include <span>
#include <omp.h>

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: "
                  << argv[0]
                  << " <binary-file>\n";
        return EXIT_FAILURE;
    }

    std::string fileName = argv[1];

    Timer timer;
    timer.start();

    long long totalItems = 0;
    double totalSum = 0.0;

#pragma omp parallel reduction(+ : totalItems, totalSum)
    {
        int id = omp_get_thread_num();
        int numThreads = omp_get_num_threads();

        ThreadReader<double> reader(fileName, id, numThreads);

        std::span<const double> chunk = reader.readChunk();

        totalItems += static_cast<long long>(chunk.size());

        // Actually read every value from the mapped file.
        double localSum = 0.0;
        for (double value : chunk)
        {
            localSum += value;
        }

        totalSum += localSum;

#pragma omp critical
        {
            std::cout << "Thread "
                      << id
                      << " read "
                      << chunk.size()
                      << " doubles\n";
        }
    }

    timer.stop();

    std::cout << "\nTotal doubles read = "
              << totalItems
              << '\n';

    std::cout << "Checksum = "
              << totalSum
              << '\n';

    std::cout << "Elapsed time = "
              << timer.getTime()
              << " seconds\n";

    return EXIT_SUCCESS;
}