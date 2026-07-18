#include <iostream>




int main(){
    MPIProcessReader<double> doubleReader;
    doubleReader.open("data/1b-doubles.bin");
    std::vector<double> doubles = doubleReader.readChunk();
    // print how many doubles were read by each process
    int rank = mpiRuntime().getRank();
    std::cout << "Process " << rank << " read " << doubles.size() << " doubles." << std::endl;
    doubleReader.close();
    return 0;
}