

/* CUDAIO.h contains the CUDAIO class that provides I/O operations using
 * NVIDIA GPUDirect Storage (GDS) for reading and writing files directly to
 * and from GPU memory.
 *
 * author: Joshua Eilu for Professor Joel Adams at Calvin University.
 * date:   Summer 2026
 */

#ifndef CUDAIO_H
#define CUDAIO_H
#include "IO_Base.h"

#include <cuda_runtime.h>
#include <cufile.h>

#include <cerrno> // errno
#include <limits> // std::numeric_limits

/********************************************************************
 * Helper function declarations.
 ********************************************************************/
inline void checkResult(cudaError_t result);
inline void checkResult(CUfileError_t result);

/*********************************************************************
 * CUDAIO is the base class for CUDAReader and CUDAWriter that uses NVIDIA
 * GPUDirect Storage (GDS) to read/write binary data directly to and from GPU memory.
 *
 * The CUDA backend currently uses one processing element: id = 0, numPEs = 1
 *
 * It's subclasses are CUDAReader and CUDAWriter.
 *
 */

template <class ItemType>
class CUDAIO : public IO_Base<ItemType> {

  public:
    CUDAIO();

    void open(const std::string &fileName, int openMode) override;
    void close() override;

    void setHandleRegistered(bool registered) { this->cfHandleRegistered = registered; }

    virtual ~CUDAIO() = default;

  protected:
    int fd = -1;

    CUfileHandle_t cfHandle{}; // cuFile handle for the open file
    bool cfHandleRegistered = false;

    void *devPtr = nullptr;
    std::size_t bufferSize = 0;

    void freeDeviceBuffer(); // Frees the GPU memory buffer if allocated
};

/* freeDeviceBuffer()
 *
 * Precondition: devPtr points to an allocated GPU memory buffer.
 *                 && bufferSize > 0
 * Postcondition: the GPU memory buffer is freed and devPtr is set to nullptr.
 */
template <class ItemType>
void CUDAIO<ItemType>::freeDeviceBuffer() {
    if (this->devPtr != nullptr) {
        checkResult(cudaFree(this->devPtr));
        this->devPtr = nullptr;
        this->bufferSize = 0;
    }
}

/*
 * CUDAIO constructor
 *
 * Postcondition: CUDAIO object is initialized with one processing element
 * (id = 0, numPEs = 1) and no file opened.
 *
 */
template <class ItemType>
CUDAIO<ItemType>::CUDAIO() : IO_Base<ItemType>(0, 1) {}

/*
 * CUDAIO::open
 * @param: fileName, path to the file
 * @param: openMode, POSIX file open flags
 * Allowed access modes: O_RDONLY, O_RDWR, O_WRONLY, O_CREAT, O_TRUNC, O_DIRECT
 * Precondition: no file is currently open with this object.
 * Postcondition: file is open
 *      & file descriptor is registered with cuFile
 *      & IO_Base file metadata is initialized
 */

template <class ItemType>
void CUDAIO<ItemType>::open(const std::string &fileName, int openMode) {
    // Validate access mode
    const int accessMode = openMode & O_ACCMODE;

    if (accessMode != O_RDONLY && accessMode != O_RDWR) {

        throw std::invalid_argument("CUDAIO::open(): expected O_RDONLY or O_RDWR");
    }

    // Prevent opening another file with this object
    if (IO_Base<ItemType>::getFileOpened()) {

        throw std::runtime_error("CUDAIO::open(): a file is already open");
    }

    // Open POSIX file
    if (openMode & O_CREAT) {

        this->fd = ::open(fileName.c_str(), openMode, 0644);

    } else {

        this->fd = ::open(fileName.c_str(), openMode);
    }

    if (this->fd < 0) {

        throw std::runtime_error(std::string("CUDAIO::open(): failed to open file: ") +
                                 std::strerror(errno));
    }

    const long itemSize = static_cast<long>(sizeof(ItemType));

    // Reader
    if (accessMode == O_RDONLY) {

        struct stat fileInfo{};

        if (fstat(this->fd, &fileInfo) == -1) {

            ::close(this->fd);

            this->fd = -1;

            throw std::runtime_error(std::string("CUDAIO::open(): fstat failed: ") +
                                     std::strerror(errno));
        }

        const long fileSize = static_cast<long>(fileInfo.st_size);

        // File must contain complete ItemType values.
        if (fileSize % itemSize != 0) {

            ::close(this->fd);

            this->fd = -1;

            throw std::runtime_error("CUDAIO::open(): file size is not "
                                     "a multiple of sizeof(ItemType)");
        }

        IO_Base<ItemType>::setFileSize(fileSize);

        IO_Base<ItemType>::setNumItemsInFile(fileSize / itemSize);
    }

    // Writer

    else {

        // CUDAWriter's constructor has already stored the
        // intended output file size.
        const long fileSize = IO_Base<ItemType>::getFileSize();

        if (fileSize < 0) {

            ::close(this->fd);

            this->fd = -1;

            throw std::runtime_error("CUDAIO::open(): invalid output file size");
        }

        // Resize the physical file to the configured output size.
        if (ftruncate(this->fd, fileSize) == -1) {

            ::close(this->fd);

            this->fd = -1;

            throw std::runtime_error(std::string("CUDAIO::open(): ftruncate failed: ") +
                                     std::strerror(errno));
        }

        // fileSize should already have been validated by
        // CUDAWriter's constructor, but keep metadata consistent.
        IO_Base<ItemType>::setNumItemsInFile(fileSize / itemSize);
    }

    // Register file descriptor with cuFile

    CUfileDescr_t cfDescr{};

    cfDescr.handle.fd = this->fd;
    cfDescr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileError_t status = cuFileHandleRegister(&(this->cfHandle), &cfDescr);

    if (status.err != CU_FILE_SUCCESS) {

        ::close(this->fd);

        this->fd = -1;

        throw std::runtime_error("CUDAIO::open(): "
                                 "cuFileHandleRegister failed with error " +
                                 std::to_string(status.err));
    }

    setHandleRegistered(true);

    IO_Base<ItemType>::setFileName(fileName);

    IO_Base<ItemType>::setFileOpened(true);
}
/* CUDAIO::close
 * Closes the file, deregisters the cuFile handle, and frees any GPU memory
 * allocated by a CUDAReader.
 */

template <class ItemType>
void CUDAIO<ItemType>::close() {

    // Free owned GPU memory
    this->freeDeviceBuffer();

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

/*
 * CUDAReader reads an entire binary file directly into GPU memory using
 * NVIDIA GPUDirect Storage.
 */

template <class ItemType>
class CUDAReader : public CUDAIO<ItemType> {

  public:
    CUDAReader();
    CUDAReader(const std::string &fileName);
    ItemType *readToGPU(); // Reads the entire file into GPU memory

    std::vector<ItemType> readChunksToGPU(); // Reads the file in chunks into GPU memory

    template <class Callback>
    std::vector<ItemType>
    readChunksToGPU(Callback callback); // Reads the file in chunks into GPU memory and
                                        // applies the callback to each chunk

    ~CUDAReader() = default;
};

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
 * Precondition: file is open and ready for reading.
 *                  && file can fit into available GPU memory.
 * Postcondition: GPU memory contains the entire file content, or nullptr if the file is
 * empty.
 */

template <class ItemType>
ItemType *CUDAReader<ItemType>::readToGPU() {

    // Make sure a file has been opened
    if (!IO_Base<ItemType>::getFileOpened()) {

        throw std::runtime_error("CUDAReader::readToGPU(): file not opened");
    }

    const std::size_t fileSize =
        static_cast<std::size_t>(IO_Base<ItemType>::getFileSize());

    if (fileSize == 0) {
        // If a previous read allocated memory, release it.
        this->freeDeviceBuffer();

        return nullptr;
    }

    // Release buffer from a previous readToGPU() call
    this->freeDeviceBuffer();

    std::size_t freeMemory = 0;
    std::size_t totalMemory = 0;
    cudaError_t cudaStatus = cudaMemGetInfo(&freeMemory, &totalMemory);
    checkResult(cudaStatus);

    if (fileSize > freeMemory) {
        throw std::runtime_error(
            "CUDAReader::readToGPU(): file is too large to fit in available GPU memory. "
            "Required: " +
            std::to_string(fileSize) +
            " bytes, available: " + std::to_string(freeMemory) +
            " bytes. Use readChunksToGPU() to read files larger than available GPU "
            "memory.");
    }

    // Allocate GPU memory
    checkResult(cudaMalloc(&(this->devPtr), fileSize));

    this->bufferSize = fileSize;

    // Read directly from storage into GPU memory
    ssize_t readBytes = cuFileRead(this->cfHandle, this->devPtr, this->bufferSize, 0, 0);

    // Check for GDS read failure
    if (readBytes < 0) {
        this->freeDeviceBuffer();
        throw std::runtime_error(
            "CUDAReader::readToGPU(): cuFileRead failed with error " +
            std::to_string(readBytes));
    }

    // Verify that the entire file was read
    if (static_cast<std::size_t>(readBytes) != fileSize) {
        const std::size_t actualBytes = static_cast<std::size_t>(readBytes);
        this->freeDeviceBuffer();
        throw std::runtime_error("CUDAReader::readToGPU(): read bytes mismatch. "
                                 "Expected: " +
                                 std::to_string(fileSize) + " bytes, read: " +
                                 std::to_string(actualBytes) + " bytes");
    }

    return static_cast<ItemType *>(this->devPtr);
}

/*
 * CUDAReader::readChunksToGPU
 * Reads the file in chunks that can fit into available GPU memory.
 *
 * Precondition: file is open and ready for reading.
 * @note: This function is useful for reading large files that cannot fit entirely into
 * GPU memory. Postcondition: Host memory contains the entire file content in a contiguous
 * vector.
 */
template <class ItemType>
std::vector<ItemType> CUDAReader<ItemType>::readChunksToGPU() {

    return readChunksToGPU([](ItemType *, std::size_t) {
        // No processing
    });
}

/**
 * CUDAReader::readChunksToGPU with a callback
 * Reads the file in chunks that can fit into available GPU memory and processes each
 * chunk using the provided callback.
 * @param processChunk: A callback function that takes a pointer to GPU memory and the
 * number of items in the chunk.
 *
 * Precondition: file is open and ready for reading
 * @note: This function is useful for reading, and processing large files that cannot fit
 * entirely into GPU memory. Postcondition: Host memory contains the entire file content
 * in a contiguous vector.
 */

template <class ItemType>
template <class Callback>
std::vector<ItemType> CUDAReader<ItemType>::readChunksToGPU(Callback processChunk) {

    if (!IO_Base<ItemType>::getFileOpened()) {
        throw std::runtime_error("CUDAReader::readChunksToGPU(): file not opened");
    }

    const std::size_t fileSize =
        static_cast<std::size_t>(IO_Base<ItemType>::getFileSize());

    const std::size_t totalItems = fileSize / sizeof(ItemType);

    if (fileSize == 0) {
        return {};
    }

    this->freeDeviceBuffer();

    // Find available GPU memory
    std::size_t freeMemory = 0;
    std::size_t totalMemory = 0;

    checkResult(cudaMemGetInfo(&freeMemory, &totalMemory));

    constexpr double MEMORY_FRACTION = 0.90;

    std::size_t availableSize = static_cast<std::size_t>(freeMemory * MEMORY_FRACTION);

    this->bufferSize = std::min(fileSize, availableSize);

    // Keep complete ItemType values
    this->bufferSize -= this->bufferSize % sizeof(ItemType);

    if (this->bufferSize == 0) {
        throw std::runtime_error("CUDAReader::readChunksToGPU(): "
                                 "not enough available GPU memory");
    }

    // Allocate ONE reusable GPU buffer
    checkResult(cudaMalloc(&(this->devPtr), this->bufferSize));

    // Host vector will eventually contain the entire file
    std::vector<ItemType> fileData(totalItems);

    std::size_t fileOffset = 0;
    std::size_t itemOffset = 0;

    while (fileOffset < fileSize) {

        const std::size_t remainingBytes = fileSize - fileOffset;

        const std::size_t bytesToRead = std::min(this->bufferSize, remainingBytes);

        ssize_t readBytes = cuFileRead(this->cfHandle, this->devPtr, bytesToRead,
                                       static_cast<off_t>(fileOffset), 0);

        if (readBytes < 0) {
            this->freeDeviceBuffer();

            throw std::runtime_error("CUDAReader::readChunksToGPU(): "
                                     "cuFileRead failed");
        }

        if (readBytes == 0) {
            this->freeDeviceBuffer();

            throw std::runtime_error("CUDAReader::readChunksToGPU(): "
                                     "unexpected end of file");
        }

        const std::size_t numItems =
            static_cast<std::size_t>(readBytes) / sizeof(ItemType);

        ItemType *gpuData = static_cast<ItemType *>(this->devPtr);

        // ---------------------------------
        // Process current chunk on the GPU
        // ---------------------------------

        processChunk(gpuData, numItems);

        checkResult(cudaDeviceSynchronize());

        // ---------------------------------
        // Preserve this chunk on the CPU
        // before devPtr gets overwritten
        // ---------------------------------

        checkResult(cudaMemcpy(fileData.data() + itemOffset, gpuData,
                               static_cast<std::size_t>(readBytes),
                               cudaMemcpyDeviceToHost));

        fileOffset += static_cast<std::size_t>(readBytes);

        itemOffset += numItems;
    }

    return fileData;
}

/*
 * CUDAWriter writes data that already resides in GPU memory directly to
 * storage using NVIDIA GPUDirect Storage.
 */

template <class ItemType>
class CUDAWriter : public CUDAIO<ItemType> {

  public:
    CUDAWriter(long fileSize);

    CUDAWriter(long fileSize, const std::string &fileName);

    // Writes numItems values directly from GPU memory to the file.
    void writeToFile(const ItemType *gpuData, std::size_t numItems);

    ~CUDAWriter() = default;
};

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
    CUDAIO<ItemType>::open(fileName, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT);
}

/*
 * CUDAWriter::writeToFile
 *
 * Writes data directly from GPU memory to storage.
 *
 * @param: gpuData, pointer to GPU memory
 * @param: numItems, number of ItemType objects stored at gpuData
 *
 * Precondition: output file has been opened
 *      - gpuData points to CUDA device or managed memory
 *      - numItems corresponds to the configured output file size
 *
 */

template <class ItemType>
void CUDAWriter<ItemType>::writeToFile(const ItemType *gpuData, std::size_t numItems) {

    if (!IO_Base<ItemType>::getFileOpened()) {

        throw std::runtime_error("CUDAWriter::writeToFile(): file not opened");
    }

    const std::size_t bytesToWrite = numItems * sizeof(ItemType);

    const std::size_t expectedBytes =
        static_cast<std::size_t>(IO_Base<ItemType>::getFileSize());

    if (bytesToWrite == 0) {

        if (expectedBytes != 0) {
            throw std::invalid_argument(
                "CUDAWriter::writeToFile(): zero items supplied for a non-empty file");
        }

        return;
    }

    if (gpuData == nullptr) {

        throw std::invalid_argument("CUDAWriter::writeToFile(): gpuData is nullptr");
    }

    // The current CUDA backend has one PE, so one chunk is the whole file
    if (bytesToWrite != expectedBytes) {

        throw std::invalid_argument(
            "CUDAWriter::writeToFile(): GPU buffer size does not match output file size. "
            "Expected: " +
            std::to_string(expectedBytes) +
            " bytes, received: " + std::to_string(bytesToWrite) + " bytes");
    }

    // Verify that the pointer refers to CUDA-accessible memory
    cudaPointerAttributes attributes{};

    checkResult(cudaPointerGetAttributes(&attributes, gpuData));

    if (attributes.type != cudaMemoryTypeDevice) {
        throw std::invalid_argument(
            "CUDAWriter::writeToFile(): expected CUDA device memory");
    }

    // Write directly from GPU memory to storage
    ssize_t writtenBytes = cuFileWrite(this->cfHandle, gpuData, bytesToWrite, 0, 0);

    // Check for GDS write failure
    if (writtenBytes < 0) {
        throw std::runtime_error(
            "CUDAWriter::writeToFile(): cuFileWrite failed with error " +
            std::to_string(writtenBytes));
    }

    // Verify complete write
    if (static_cast<std::size_t>(writtenBytes) != bytesToWrite) {
        throw std::runtime_error("CUDAWriter::writeToFile(): write bytes mismatch. "
                                 "Expected: " +
                                 std::to_string(bytesToWrite) + " bytes, wrote: " +
                                 std::to_string(writtenBytes) + " bytes");
    }

    // Flush file changes
    if (fsync(this->fd) != 0) {

        throw std::runtime_error(
            std::string("CUDAWriter::writeToFile(): fsync failed: ") +
            std::strerror(errno));
    }
}

// Helper Utilities

inline void checkResult(cudaError_t result) {

    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Error: ") +
                                 cudaGetErrorString(result));
    }
}

inline void checkResult(CUfileError_t result) {

    if (result.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("cuFile Error: " + std::to_string(result.err));
    }
}
#endif // CUDAIO_H
