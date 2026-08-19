#include "OO_IO/include/OO_MPI_IO.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, processCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    MPIProcessReader<double> reader(rank, processCount, "./tests/files/empty.bin");

    std::vector<double> chunk = reader.readChunk();

    reader.close();

    MPI_Finalize();
    return 0;
}