#include "OO_IO.h"

#include <iostream>
#include <span>
#include <omp.h>

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <binary-file>\n";
        return EXIT_FAILURE;
    }

    std::string fileName = argv[1];

    const int numRuns = 3;
    const int numThreads = 12;

    for (int run = 1; run <= numRuns; ++run)
    {
        Timer timer;
        long long totalItems = 0;
        double totalSum = 0.0;

        timer.start();

#pragma omp parallel num_threads(numThreads) reduction(+ : totalItems, totalSum)
        {
            int id = omp_get_thread_num();

            ThreadReader<double> reader(fileName, id, numThreads);

            std::span<const double> chunk = reader.readChunk();

            totalItems += static_cast<long long>(chunk.size());

            double localSum = 0.0;
            for (double value : chunk)
            {
                localSum += value;
            }

            totalSum += localSum;
        }

        timer.stop();

        std::cout << "Run " << run << ":\n";
        std::cout << "Total doubles read = " << totalItems << '\n';
        std::cout << "Checksum = " << totalSum << '\n';
        std::cout << "Elapsed time = " << timer.getTime() << " seconds\n\n";
    }

    return EXIT_SUCCESS;
}