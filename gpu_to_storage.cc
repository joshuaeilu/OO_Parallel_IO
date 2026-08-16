#include <iostream>
#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cufile.h>


int main() {

    CUfileHandle_t cfHandle;
    CUfileDescr_t cfDescr = {};

    const char* filename = "./written_files/1b-doubles.bin";

    const size_t numDoubles = 1'000'000'000ULL;
    const size_t bufferSize = numDoubles * sizeof(double);

    std::cout << "Number of doubles: " << numDoubles << '\n';
    std::cout << "Total bytes: " << bufferSize << '\n';


    // -------------------------------------------------------------------------
    // Check available GPU memory
    // -------------------------------------------------------------------------

    size_t freeMemory;
    size_t totalMemory;

    cudaError_t cudaStatus =
        cudaMemGetInfo(&freeMemory, &totalMemory);

    if (cudaStatus != cudaSuccess) {
        std::cerr
            << "cudaMemGetInfo failed: "
            << cudaGetErrorString(cudaStatus)
            << '\n';

        return 1;
    }

    std::cout
        << "GPU memory available: "
        << freeMemory
        << " bytes\n";

    std::cout
        << "GPU memory total: "
        << totalMemory
        << " bytes\n";


    if (bufferSize > freeMemory) {

        std::cerr
            << "\nNot enough free GPU memory.\n"
            << "Required: "
            << bufferSize
            << " bytes\n"
            << "Available: "
            << freeMemory
            << " bytes\n";

        return 1;
    }


    // -------------------------------------------------------------------------
    // Open file
    // -------------------------------------------------------------------------

    int fd = open(
        filename,
        O_CREAT | O_RDWR | O_TRUNC | O_DIRECT,
        0664
    );

    if (fd < 0) {
        perror("File open failed");
        return 1;
    }


    // -------------------------------------------------------------------------
    // Set up GDS descriptor
    // -------------------------------------------------------------------------

    cfDescr.handle.fd = fd;
    cfDescr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileError_t status =
        cuFileHandleRegister(
            &cfHandle,
            &cfDescr
        );


    if (status.err != CU_FILE_SUCCESS) {

        std::cerr
            << "cuFileHandleRegister failed: "
            << status.err
            << '\n';

        close(fd);

        return 1;
    }


    // -------------------------------------------------------------------------
    // Allocate GPU memory
    // -------------------------------------------------------------------------

    void* devPtr = nullptr;

    std::cout
        << "\nAllocating "
        << bufferSize
        << " bytes on GPU...\n";


    cudaStatus =
        cudaMalloc(
            &devPtr,
            bufferSize
        );


    if (cudaStatus != cudaSuccess) {

        std::cerr
            << "cudaMalloc failed: "
            << cudaGetErrorString(cudaStatus)
            << '\n';

        cuFileHandleDeregister(cfHandle);
        close(fd);

        return 1;
    }


    std::cout
        << "GPU allocation successful.\n";


    // -------------------------------------------------------------------------
    // Initialize all doubles to 0.0
    // -------------------------------------------------------------------------

    cudaStatus =
        cudaMemset(
            devPtr,
            0,
            bufferSize
        );


    if (cudaStatus != cudaSuccess) {

        std::cerr
            << "cudaMemset failed: "
            << cudaGetErrorString(cudaStatus)
            << '\n';

        cudaFree(devPtr);
        cuFileHandleDeregister(cfHandle);
        close(fd);

        return 1;
    }


    // -------------------------------------------------------------------------
    // Write GPU memory directly to disk
    // -------------------------------------------------------------------------

    std::cout
        << "Writing GPU buffer to disk...\n";


    ssize_t writtenBytes =
        cuFileWrite(
            cfHandle,
            devPtr,
            bufferSize,
            0,
            0
        );


    if (writtenBytes < 0) {

        std::cerr
            << "cuFileWrite failed: "
            << writtenBytes
            << '\n';

    } else {

        std::cout
            << "Wrote "
            << writtenBytes
            << " bytes to disk.\n";
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


    std::cout
        << "Finished writing "
        << filename
        << '\n';


    return 0;
}