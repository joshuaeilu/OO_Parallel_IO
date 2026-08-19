/*
 * mpi_write_example.cpp
 *
 * Example of writing genome data using the
 * OO_Parallel_IO MPI backend.
 *
 * Build: mpic++ -std=c++20 -o mpi_write_example mpi_write_example.cpp
 * Run:   mpirun -np <num_processes> ./mpi_write_example
 *
 * @author: Josh Eilu
 * @date: Summer 2026
 */

#include "../OO_IO/include/OO_MPI_IO.h"

#include <mpi.h>

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numProcesses = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    // Create data for this process
    const std::string genomeSequence = "ACGTACGTACGTACGT";

    std::vector<char> chunk(genomeSequence.begin(), genomeSequence.end());
    
    // Open the output file for parallel writing
    MPIProcessWriter<char> writer(rank, numProcesses);

    writer.open("../genome_output.dna", MPI_MODE_WRONLY | MPI_MODE_CREATE);

    // Write this process's chunk
    writer.writeChunk(chunk);

    std::cout << "Process " << rank << " wrote " << chunk.size() << " characters.\n";

    // Close the output file
    writer.close();

    // Shut down MPI
    MPI_Finalize();

    return 0;
}