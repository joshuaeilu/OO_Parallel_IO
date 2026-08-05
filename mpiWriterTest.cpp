#include "OO_IO/include/OO_MPI_IO.h"
#include <vector>
#include <iostream>


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPIProcessReader<double> reader(rank, size, "data/1b-doubles.bin");
    MPIProcessWriter<double> writer(rank, size, "data/copy-1b-doubles.bin");
    std::vector<double> chunk = reader.readChunk(); // Use the data read from the original file
    writer.writeChunk(chunk);
    reader.close();
    writer.close();

    MPI_Finalize();
    return 0;
}