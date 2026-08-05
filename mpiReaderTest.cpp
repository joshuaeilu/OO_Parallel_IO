
#include "OO_IO/include/OO_MPI_IO.h"
#include <vector>
#include <iostream>


int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // MPIProcessReader<double> reader(rank, size);
    // reader.open("data/1b-doubles.bin", MPI_MODE_RDONLY);
    // std::vector<double> chunk = reader.readChunk();
    // std::cout << "Rank " << rank << " read " << chunk.size() << " items." << std::endl;
    // reader.close();

    MPIProcessReader<double> reader(rank, size, "data/1b-dobles.bin");
    std::vector<double> chunk = reader.readChunk();
    std::cout << "Rank " << rank << " read " << chunk.size() << " items." << std::endl;
    reader.close();
    
    MPI_Finalize();
    return 0;
}