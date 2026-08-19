#ifndef CUDAIO_H
#define CUDAIO_H

/* CUDAIO.h contains the CUDAIO class that provides I/O operations using
 * NVIDIA GPUDirect Storage (GDS) for reading and writing files directly to
 * and from GPU memory.
 *
 * author: Joshua Eilu for Professor Joel Adams at Calvin University.
 * date:   Summer 2026
 */

#include "IO_Base.h"

#include <cuda_runtime.h>
#include <cufile.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

inline void checkResult(cudaError_t result);
inline void checkResult(CUfileError_t result);
/*
 * CUDAIO
 *
 *
 * CUDAIO is the base class for CUDAReader and CUDAWriter.
 *
 * The CUDA backend currently uses one processing element:
 *
 *      id     = 0
 *      numPEs = 1
 *
 * CUDAIO owns:
 *
 *      - the POSIX file descriptor
 *      - the cuFile handle
 *      - any GPU buffer allocated by a CUDAReader
 *
 * CUDAWriter does not take ownership of GPU pointers passed to writeChunk().
 */

template <class ItemType>
class CUDAIO : public IO_Base<ItemType> {

  public:
    CUDAIO();

    void open(const std::string &fileName, int openMode) override;
    void close() override;

    void setHandleRegistered(bool registered) { this->cfHandleRegistered = registered; }
    virtual ~CUDAIO();

  protected:
    int fd = -1;

    CUfileHandle_t cfHandle{};

    bool cfHandleRegistered = false;

    void *devPtr = nullptr;

    std::size_t bufferSize = 0;
};

/*
 * CUDAReader
 *
 *
 * CUDAReader reads an entire binary file directly into GPU memory using
 * NVIDIA GPUDirect Storage.
 */

template <class ItemType>
class CUDAReader : public CUDAIO<ItemType> {

  public:
    CUDAReader();
    CUDAReader(const std::string &fileName);
    ItemType *readToGPU();

    ~CUDAReader() = default;
};

/*
 * CUDAWriter
 *
 *
 * CUDAWriter writes data that already resides in GPU memory directly to
 * storage using NVIDIA GPUDirect Storage.
 */

template <class ItemType>
class CUDAWriter : public CUDAIO<ItemType> {

  public:
    CUDAWriter(long fileSize);

    CUDAWriter(long fileSize, const std::string &fileName);

    /*
     * CUDAWriter needs slightly different open behavior because the output
     * file must be resized to the requested file size.
     */
    void open(const std::string &fileName, int openMode) override;

    /*
     * Writes numItems values directly from GPU memory to the file.
     */
    void writeChunk(const ItemType *gpuData, std::size_t numItems);

    ~CUDAWriter() = default;
};

/*
 * CUDAIO constructor
 *
 */

template <class ItemType>
CUDAIO<ItemType>::CUDAIO() : IO_Base<ItemType>(0, 1) {}

/*
 * CUDAIO::open
 *
 * @param: fileName, path to the file
 * @param: openMode, POSIX file open flags
 *
 * Allowed access modes: O_RDONLY, O_RDWR, O_WRONLY, O_CREAT, O_TRUNC, O_DIRECT
 *
 * Postcondition: file is open
 *      & file descriptor is registered with cuFile
 *      & IO_Base file metadata is initialized
 */

template <class ItemType>
void CUDAIO<ItemType>::open(const std::string &fileName, int openMode) {
    // Ensure that the correct access mode is specified
    int accessMode = openMode & O_ACCMODE;
    if (accessMode != O_RDONLY && accessMode != O_RDWR) {

        throw std::invalid_argument("CUDAIO::open(): expected O_RDONLY or O_RDWR");
    }

    // Prevent opening a second file with the same object
    if (IO_Base<ItemType>::getFileOpened()) {
        throw std::runtime_error("CUDAIO::open(): a file is already open");
    }

    // Open POSIX file
    if (openMode & O_CREAT) {

        this->fd = ::open(fileName.c_str(), openMode, 0644); // if file does not exist

    } else {

        this->fd = ::open(fileName.c_str(), openMode); // if file exists
    }

    if (this->fd < 0) {

        throw std::runtime_error(std::string("CUDAIO::open(): failed to open file: ") +
                                 std::strerror(errno));
    }

    // Register file descriptor with cuFile
    CUfileDescr_t cfDescr{};
    cfDescr.handle.fd = this->fd;
    cfDescr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileError_t status = cuFileHandleRegister(&(this->cfHandle), &cfDescr);
    checkResult(status);

    setHandleRegistered(true);

    // get file information
    struct stat fileInfo{};
    if (fstat(this->fd, &fileInfo) == -1) {
        cuFileHandleDeregister(this->cfHandle);
        setHandleRegistered(false);
        ::close(this->fd);
        this->fd = -1;
        throw std::runtime_error(std::string("CUDAIO::open(): fstat failed: ") +
                                 std::strerror(errno));
    }

    long fileSize = static_cast<long>(fileInfo.st_size);

    const long itemSize = static_cast<long>(sizeof(ItemType));
    IO_Base<ItemType>::setFileName(fileName);
    IO_Base<ItemType>::setFileSize(fileSize);
    IO_Base<ItemType>::setNumItemsInFile(fileSize / itemSize);
    IO_Base<ItemType>::setFileOpened(true);
}

/* CUDAIO::close
 * Closes the file, deregisters the cuFile handle, and frees any GPU memory
 * allocated by a CUDAReader.
 */

template <class ItemType>
void CUDAIO<ItemType>::close() {

    // Free owned GPU memory
    if (this->devPtr != nullptr) {
        cudaFree(this->devPtr);
        this->devPtr = nullptr;
        this->bufferSize = 0;
    }

    // Deregister cuFile handle
    if (this->cfHandleRegistered) {

        cuFileHandleDeregister(this->cfHandle);
        setHandleRegistered(false);
    }

    // Close POSIX file descriptor
    if (this->fd >= 0) {
        ::close(this->fd);
        this->fd = -1;
    }

    IO_Base<ItemType>::setFileOpened(false);
}

/* CUDAReader minimal constructor
 */

template <class ItemType>
CUDAReader<ItemType>::CUDAReader() : CUDAIO<ItemType>() {}

/*
 * CUDAReader file-based constructor
 * @param: fileName, path to the file
 * Precondition: fileName is a valid path to an existing file
 * Postcondition: file is open and ready for reading
 *
 */

template <class ItemType>
CUDAReader<ItemType>::CUDAReader(const std::string &fileName) : CUDAIO<ItemType>() {
    this->open(fileName, O_RDONLY | O_DIRECT);
}

/* CUDAReader::readToGPU
 * Reads the entire file directly into GPU memory.
 *
 * The returned pointer is owned by this CUDAReader.
 *
 * The caller must NOT call cudaFree() on the returned pointer.
 * CUDAReader::close() or the destructor releases the GPU memory.
 *
 * Returns nullptr for an empty file.
 */

template <class ItemType>
ItemType *CUDAReader<ItemType>::readToGPU() {

    // Make sure a file has been opened
    if (!IO_Base<ItemType>::getFileOpened()) {

        throw std::runtime_error("CUDAReader::readToGPU(): file not opened");
    }

    const std::size_t fileSize = static_cast<std::size_t>(IO_Base<ItemType>::getFileSize());

    if (fileSize == 0) {
        // If a previous read allocated memory, release it.
        if (this->devPtr != nullptr) {
            checkResult(cudaFree(this->devPtr));
            this->devPtr = nullptr;
        }
        this->bufferSize = 0;

        return nullptr;
    }

    // Release buffer from a previous readToGPU() call
    if (this->devPtr != nullptr) {
        checkResult(cudaFree(this->devPtr));
        this->devPtr = nullptr;
        this->bufferSize = 0;
    }

    std::size_t freeMemory = 0;
    std::size_t totalMemory = 0;
    cudaError_t cudaStatus = cudaMemGetInfo(&freeMemory, &totalMemory);
    checkResult(cudaStatus);

    if (fileSize > freeMemory) {
        throw std::runtime_error(
            "CUDAReader::readToGPU(): file is too large to fit in available GPU memory. "
            "Required: " +
            std::to_string(fileSize) + " bytes, available: " + std::to_string(freeMemory) +
            " bytes, total GPU memory: " + std::to_string(totalMemory) + " bytes");
    }

    // Allocate GPU memory
    checkResult(cudaMalloc(&(this->devPtr), fileSize));

    this->bufferSize = fileSize;

    // Read directly from storage into GPU memory
    ssize_t readBytes = cuFileRead(this->cfHandle, this->devPtr, this->bufferSize, 0, 0);

    // Check for GDS read failure
    if (readBytes < 0) {
        checkResult(cudaFree(this->devPtr));
        this->devPtr = nullptr;
        this->bufferSize = 0;
        throw std::runtime_error("CUDAReader::readToGPU(): cuFileRead failed with error " +
                                 std::to_string(readBytes));
    }

    // Verify that the entire file was read
    if (static_cast<std::size_t>(readBytes) != fileSize) {
        const std::size_t actualBytes = static_cast<std::size_t>(readBytes);
        checkResult(cudaFree(this->devPtr));
        this->devPtr = nullptr;
        this->bufferSize = 0;
        throw std::runtime_error("CUDAReader::readToGPU(): read bytes mismatch. "
                                 "Expected: " +
                                 std::to_string(fileSize) +
                                 " bytes, read: " + std::to_string(actualBytes) + " bytes");
    }

    return static_cast<ItemType *>(this->devPtr);
}

/*
 * CUDAWriter minimal constructor
 * @param: fileSize, desired output file size in bytes
 */

template <class ItemType>
CUDAWriter<ItemType>::CUDAWriter(long fileSize) : CUDAIO<ItemType>() {

    if (fileSize < 0) {
        throw std::invalid_argument("CUDAWriter(): fileSize cannot be negative");
    }

    const long itemSize = static_cast<long>(sizeof(ItemType));

    if (fileSize % itemSize != 0) {
        throw std::invalid_argument(
            "CUDAWriter(): fileSize must be a multiple of sizeof(ItemType)");
    }

    IO_Base<ItemType>::setFileSize(fileSize);
    IO_Base<ItemType>::setNumItemsInFile(fileSize / itemSize);
}

/*
 * CUDAWriter file-based constructor
 *
 */

template <class ItemType>
CUDAWriter<ItemType>::CUDAWriter(long fileSize, const std::string &fileName)
    : CUDAWriter<ItemType>(fileSize) {
    this->open(fileName, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT);
}

/*
 * CUDAWriter::writeChunk
 *
 * Writes data directly from GPU memory to storage.
 *
 * @param: gpuData, pointer to GPU memory
 *
 * @param: numItems, number of ItemType objects stored at gpuData
 *
 * Precondition: output file has been opened
 *      - gpuData points to CUDA device or managed memory
 *      - numItems corresponds to the configured output file size
 *
 */

template <class ItemType>
void CUDAWriter<ItemType>::writeChunk(const ItemType *gpuData, std::size_t numItems) {

    if (!IO_Base<ItemType>::getFileOpened()) {

        throw std::runtime_error("CUDAWriter::writeChunk(): file not opened");
    }

    const std::size_t bytesToWrite = numItems * sizeof(ItemType);

    const std::size_t expectedBytes = static_cast<std::size_t>(IO_Base<ItemType>::getFileSize());

    if (bytesToWrite == 0) {

        if (expectedBytes != 0) {

            throw std::invalid_argument(
                "CUDAWriter::writeChunk(): zero items supplied for a non-empty file");
        }

        return;
    }

    if (gpuData == nullptr) {

        throw std::invalid_argument("CUDAWriter::writeChunk(): gpuData is nullptr");
    }

    // The current CUDA backend has one PE, so one chunk is the whole file
    if (bytesToWrite != expectedBytes) {

        throw std::invalid_argument(
            "CUDAWriter::writeChunk(): GPU buffer size does not match output file size. "
            "Expected: " +
            std::to_string(expectedBytes) + " bytes, received: " + std::to_string(bytesToWrite) +
            " bytes");
    }

    // Verify that the pointer refers to CUDA-accessible memory
    cudaPointerAttributes attributes{};

    checkResult(cudaPointerGetAttributes(&attributes, gpuData));

    if (attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged) {

        throw std::invalid_argument(
            "CUDAWriter::writeChunk(): expected CUDA device or managed memory");
    }

    // Write directly from GPU memory to storage
    ssize_t writtenBytes = cuFileWrite(this->cfHandle, gpuData, bytesToWrite, 0, 0);

    // Check for GDS write failure
    if (writtenBytes < 0) {

        throw std::runtime_error("CUDAWriter::writeChunk(): cuFileWrite failed with error " +
                                 std::to_string(writtenBytes));
    }

    // Verify complete write

    if (static_cast<std::size_t>(writtenBytes) != bytesToWrite) {

        throw std::runtime_error("CUDAWriter::writeChunk(): write bytes mismatch. "
                                 "Expected: " +
                                 std::to_string(bytesToWrite) +
                                 " bytes, wrote: " + std::to_string(writtenBytes) + " bytes");
    }

    // Flush file changes
    if (fsync(this->fd) != 0) {

        throw std::runtime_error(std::string("CUDAWriter::writeChunk(): fsync failed: ") +
                                 std::strerror(errno));
    }
}

// Helper Utilities
 
inline void checkResult(cudaError_t result) {

    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA Error: ") +
            cudaGetErrorString(result)
        );
    }
}


inline void checkResult(CUfileError_t result) {

    if (result.err != CU_FILE_SUCCESS) {
        throw std::runtime_error(
            "cuFile Error: " +
            std::to_string(result.err)
        );
    }
}
#endif  // CUDAIO_H
