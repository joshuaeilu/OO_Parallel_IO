#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <cuda_runtime.h>
#include <cufile.h>

int main() {

    CUfileHandle_t cfHandle;    // Get a handle to the file for GDS operations
    CUfileDescr_t cfDescr = {}; // Initialize the descriptor to zero

    const std::size_t numDoubles = 1'000'000ULL;
    std::vector<double> data(numDoubles);

    std::ifstream inputFile("./files/1m-doubles.bin", std::ios::binary);
    if (!inputFile) {
        std::cerr << "Failed to open file\n";
        return 1;
    }

    inputFile.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(double)));

    if (inputFile.gcount() != static_cast<std::streamsize>(data.size() * sizeof(double))) {
        std::cerr << "Could not read the full file\n";
        return 1;
    }



    // -------------------------------------------------------------------------
    // Check available GPU memory
    // -------------------------------------------------------------------------

    size_t freeMemory;
    size_t totalMemory;

    cudaError_t cudaStatus = cudaMemGetInfo(&freeMemory, &totalMemory);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(cudaStatus) << '\n';

        return 1;
    }

    std::cout << "GPU memory available: " << freeMemory << " bytes\n";

    std::cout << "GPU memory total: " << totalMemory << " bytes\n";

    size_t bufferSize = data.size() * sizeof(double);

    if (bufferSize > freeMemory) {

        std::cerr << "\nNot enough free GPU memory.\n"
                  << "Required: " << bufferSize << " bytes\n"
                  << "Available: " << freeMemory << " bytes\n";

        return 1;
    }

    // -------------------------------------------------------------------------
    // Open file
    // -------------------------------------------------------------------------

    const char *filename = "./files/output.bin";
    int fd = open(filename, O_CREAT | O_RDWR | O_TRUNC | O_DIRECT, 0664);

    if (fd < 0) {
        perror("File open failed");
        return 1;
    }

    // -------------------------------------------------------------------------
    // Set up GDS descriptor
    // -------------------------------------------------------------------------

    cfDescr.handle.fd = fd;
    cfDescr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileError_t status = cuFileHandleRegister(&cfHandle, &cfDescr);

    if (status.err != CU_FILE_SUCCESS) {

        std::cerr << "cuFileHandleRegister failed: " << status.err << '\n';
        close(fd);
        return 1;
    }

    // -------------------------------------------------------------------------
    // Allocate GPU memory
    // -------------------------------------------------------------------------

    void *devPtr = nullptr;

    std::cout << "\nAllocating " << bufferSize << " bytes on GPU...\n";

    // Allocate GPU memory
    cudaStatus = cudaMalloc(&devPtr, bufferSize);

    if (cudaStatus != cudaSuccess) {

        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << '\n';

        cuFileHandleDeregister(cfHandle);
        close(fd);

        return 1;
    }

    std::cout << "GPU allocation successful.\n";

    // -------------------------------------------------------------------------
    // Initialize all doubles to 0.0
    // -------------------------------------------------------------------------

    cudaStatus = cudaMemset(devPtr, 0, bufferSize);

    if (cudaStatus != cudaSuccess) {

        std::cerr << "cudaMemset failed: " << cudaGetErrorString(cudaStatus) << '\n';

        cudaFree(devPtr);
        cuFileHandleDeregister(cfHandle);
        close(fd);

        return 1;
    }

    // -------------------------------------------------------------------------
    // Write GPU memory directly to disk
    // -------------------------------------------------------------------------

    std::cout << "Writing GPU buffer to disk...\n";

    ssize_t writtenBytes = cuFileWrite(cfHandle, devPtr, bufferSize, 0, 0);

    if (writtenBytes < 0) {

        std::cerr << "cuFileWrite failed: " << writtenBytes << '\n';

    } else {

        std::cout << "Wrote " << writtenBytes << " bytes to disk.\n";
    }

    // -------------------------------------------------------------------------
    // Flush file
    // -------------------------------------------------------------------------

    if (fsync(fd) != 0) {
        perror("fsync failed");
    }

    // -------------------------------------------------------------------------
    // Clean up
    // -------------------------------------------------------------------------

    cuFileHandleDeregister(cfHandle);

    close(fd);

    cudaFree(devPtr);

    std::cout << "Finished writing " << filename << '\n';

    return 0;
}