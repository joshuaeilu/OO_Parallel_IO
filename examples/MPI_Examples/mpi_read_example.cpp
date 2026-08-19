/*
 * mpi_read_example.cpp
 *
 * Example of reading a genome file using the
 * OO_Parallel_IO MPI backend.
 *
 * Build: mpic++ -o mpi_read_example mpi_read_example.cpp
 * Run: mpirun -np <num_processes> ./mpi_read_example
 *
 * @author: Josh Eilu
 * @date: Summer 2026
 */

#include "../OO_IO/include/OO_MPI_IO.h"

int main(int argc, char **argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numProcesses = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    // Open the genome file for parallel reading
    MPIProcessReader<char> reader(rank, numProcesses);

    reader.open("../data/Felis_catus_9.dna.plain", MPI_MODE_RDONLY);

    // Read this process's portion of the genome
    std::vector<char> chunk = reader.readChunk();

    // Process the chunk
    std::cout << "Process " << rank << " read " << chunk.size() << " characters.\n";

    // Close the file
    reader.close();

    // Shut down MPI
    MPI_Finalize();

    return 0;
}