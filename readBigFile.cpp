#include "OO_IO.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <span>
#include <string>
#include <omp.h>

volatile double checksumSink = 0.0;

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <binary-file> <num-threads>\n";
        return EXIT_FAILURE;
    }

    std::string fileName = argv[1];

    const int numRuns = 3;
    const int numThreads = std::stoi(argv[2]);

    std::string resultsFileName = "readBigFileResults.tsv";

    bool fileAlreadyExists = std::filesystem::exists(resultsFileName);

    std::ofstream resultsFile(resultsFileName, std::ios::app);

    if (!resultsFile)
    {
        std::cerr << "Error: could not open results file.\n";
        return EXIT_FAILURE;
    }

    if (!fileAlreadyExists)
    {
        resultsFile << "Run\tThreads\tElapsedSeconds\n";
    }

    resultsFile << "\nUsing OO_IO.h\n\n";

    resultsFile << std::left
              << std::setw(10) << "Run"
              << std::setw(12) << "Threads"
              << std::setw(20) << "ElapsedSeconds"
              << '\n';

    resultsFile << "------------------------------------------\n";

    for (int run = 1; run <= numRuns; ++run)
    {
        system("sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'");

        Timer timer;
        double totalSum = 0.0;

        timer.start();

#pragma omp parallel num_threads(numThreads) reduction(+ : totalSum)
        {
            int id = omp_get_thread_num();

            ThreadReader<double> reader(fileName, id, numThreads);

            std::span<const double> chunk = reader.readChunk();

            double localSum = 0.0;

            for (double value : chunk)
            {
                localSum += value;
            }

            totalSum += localSum;
            reader.close();
        }

        timer.stop();

        checksumSink = totalSum;

        resultsFile << run << '\t'
                    << numThreads << '\t'
                    << timer.getTime() << '\n';
    }

    resultsFile.close();

    return EXIT_SUCCESS;
}