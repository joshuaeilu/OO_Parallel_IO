#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

#include "OO_IO/include/ThreadsIO.h"

int main(int argc, char* argv[]) {
   if (argc < 3 || argc > 4) {
      std::cerr << "Usage: " << argv[0]
                << " <inputFile> <outputFile> [numThreads]\n";
      return EXIT_FAILURE;
   }

   std::string inputFile = argv[1];
   std::string outputFile = argv[2];
   int numThreads = argc == 4 ? std::stoi(argv[3]) : 4;

   if (numThreads <= 0) {
      std::cerr << "Error: numThreads must be positive.\n";
      return EXIT_FAILURE;
   }

Timer totalTimer;
Timer readTimer;
Timer writeTimer;

#pragma omp parallel num_threads(numThreads)
   {
      totalTimer.start();
      readTimer.start();
      // Read this thread's chunk.
      ThreadReader<double> reader(omp_get_thread_num(), omp_get_num_threads());
      reader.open(inputFile);
      std::span<const double> inputChunk = reader.readChunk();

      // Copy the data because inputChunk becomes invalid after close().
      std::vector<double> values(inputChunk.begin(),inputChunk.end());

      long fileSize = reader.getFileSize();
      reader.close();
      readTimer.stop();
      writeTimer.start();

      // Write this thread's chunk to the new file.
      ThreadWriter<double> writer(omp_get_thread_num(), omp_get_num_threads(), fileSize );
      writer.open(outputFile);
      writer.writeChunk(values);
      writer.close();
      writeTimer.stop();
      totalTimer.stop();
   }

   std::cout << "Copied " << inputFile
             << " to " << outputFile << '\n';
   std::cout << "Total time: " << totalTimer.getTime() << " seconds\n";
   std::cout << "Read time: " << readTimer.getTime() << " seconds\n";
   std::cout << "Write time: " << writeTimer.getTime() << " seconds\n";

   return EXIT_SUCCESS;
}