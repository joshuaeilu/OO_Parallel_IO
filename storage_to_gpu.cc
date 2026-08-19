#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h> //
#include <vector>

#include <cuda_runtime.h>
#include <cufile.h>

int main() {

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    const std::size_t numDoubles = 1'000'000ULL;
    const std::size_t bufferSize = numDoubles * sizeof(double);

    const char *filename = "./files/output.bin";

    std::cout << "Number of doubles: " << numDoubles << '\n';
    std::cout << "Total bytes: " << bufferSize << '\n';

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

    if (bufferSize > freeMemory) {

        std::cerr << "Not enough free GPU memory.\n"
                  << "Required: " << bufferSize << " bytes\n"
                  << "Available: " << freeMemory << " bytes\n";

        return 1;
    }

    // -------------------------------------------------------------------------
    // Open file
    // -------------------------------------------------------------------------

    int fd = open(filename, O_RDONLY | O_DIRECT);

    if (fd < 0) {
        perror("File open failed");
        return 1;
    }

    // -------------------------------------------------------------------------
    // Check file size
    // -------------------------------------------------------------------------

    struct stat fileStats;

    if (fstat(fd, &fileStats) != 0) {

        perror("fstat failed");
        close(fd);
        return 1;
    }

    std::cout << "File size: " << fileStats.st_size << " bytes\n";

    if (static_cast<std::size_t>(fileStats.st_size) < bufferSize) {

        std::cerr << "File is too small.\n"
                  << "Expected at least: " << bufferSize << " bytes\n"
                  << "Actual: " << fileStats.st_size << " bytes\n";

        close(fd);
        return 1;
    }

    // -------------------------------------------------------------------------
    // Set up GDS descriptor
    // -------------------------------------------------------------------------

    CUfileHandle_t cfHandle;
    CUfileDescr_t cfDescr = {};

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

    cudaStatus = cudaMalloc(&devPtr, bufferSize);

    if (cudaStatus != cudaSuccess) {

        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << '\n';

        cuFileHandleDeregister(cfHandle);
        close(fd);

        return 1;
    }

    std::cout << "GPU allocation successful.\n";

    // -------------------------------------------------------------------------
    // Read file directly into GPU memory
    // -------------------------------------------------------------------------

    std::cout << "Reading file directly into GPU memory...\n";

    ssize_t readBytes = cuFileRead(cfHandle, devPtr, bufferSize,
                                   0, // file offset
                                   0  // GPU buffer offset
    );

    if (readBytes < 0) {

        std::cerr << "cuFileRead failed: " << readBytes << '\n';

        cudaFree(devPtr);
        cuFileHandleDeregister(cfHandle);
        close(fd);

        return 1;
    }

    std::cout << "Read " << readBytes << " bytes from disk into GPU memory.\n";

    if (static_cast<std::size_t>(readBytes) != bufferSize) {

        std::cerr << "Warning: expected " << bufferSize << " bytes but read " << readBytes
                  << " bytes.\n";
    }

    // -------------------------------------------------------------------------
    // Copy GPU data to CPU so we can verify it
    // -------------------------------------------------------------------------

    std::vector<double> data(numDoubles);

    cudaStatus = cudaMemcpy(data.data(), devPtr, bufferSize, cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess) {

        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << '\n';

        cudaFree(devPtr);
        cuFileHandleDeregister(cfHandle);
        close(fd);

        return 1;
    }

    // -------------------------------------------------------------------------
    // Print some values for verification
    // -------------------------------------------------------------------------

    std::cout << "\nFirst 10 doubles:\n";

    for (std::size_t i = 0; i < 10 && i < numDoubles; ++i) {
        std::cout << "[" << i << "] = " << data[i] << '\n';
    }

    // -------------------------------------------------------------------------
    // Clean up
    // -------------------------------------------------------------------------

    cudaFree(devPtr);

    cuFileHandleDeregister(cfHandle);

    close(fd);

    std::cout << "\nFinished reading " << filename << '\n';

    return 0;
}