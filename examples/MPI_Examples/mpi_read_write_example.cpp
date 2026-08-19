/*
 * mpi_read_write_example.cpp
 *
 * Example of reading and writing a genome file using the
 * OO_Parallel_IO MPI backend.
 *
 * Each MPI process reads its portion of the input genome file
 * and writes that portion to a new output file.
 *
 * Build: mpic++ -o mpi_read_write_example mpi_read_write_example.cpp
 * Run:   mpirun -np <num_processes> ./mpi_read_write_example
 *
 * @author: Josh Eilu
 * @date: Summer 2026
 */

#include "../../OO_IO/include/OO_MPI_IO.h"

#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

int main(int argc, char **argv) {

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numProcesses = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    const std::string inputFile = "../../data/Felis_catus_9.dna.plain";

    const std::string outputFile = "../Felis_catus_9.copy.dna.plain";

    // ---------------------------------------------------------
    // Read the genome file
    // ---------------------------------------------------------

    MPIProcessReader<char> reader(rank, numProcesses);

    reader.open(inputFile, MPI_MODE_RDONLY);

    // Each process reads its portion of the genome.
    std::vector<char> chunk = reader.readChunk();

    std::cout << "Process " << rank << " read " << chunk.size() << " characters.\n";

    reader.close();


    // ---------------------------------------------------------
    // Write the genome chunk to the new file
    // ---------------------------------------------------------

    MPIProcessWriter<char> writer(rank, numProcesses);

    writer.open(outputFile, MPI_MODE_WRONLY | MPI_MODE_CREATE);

    // Each process writes the same chunk that it read.
    writer.writeChunk(chunk);

    std::cout << "Process " << rank << " wrote " << chunk.size() << " characters.\n";

    writer.close();

    // ---------------------------------------------------------
    // Shut down MPI
    // ---------------------------------------------------------

    MPI_Finalize();

    return 0;
}