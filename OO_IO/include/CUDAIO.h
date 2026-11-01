

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
    void setDeviceBufferSize(std::size_t newSize);
    std::size_t getDeviceBufferSize() const { return this->bufferSize; }

    std::size_t getNumChunks() const;

    virtual ~CUDAIO() = default;

  protected:
    int fd = -1;

    CUfileHandle_t cfHandle{}; // cuFile handle for the open file
    bool cfHandleRegistered = false;

    void *devPtr = nullptr;
    std::size_t bufferSize = 0;

    void freeDeviceBuffer(); // Frees the GPU memory buffer if allocated
};

template <class ItemType>
std::size_t CUDAIO<ItemType>::getNumChunks() const {
    if (this->bufferSize == 0) {
        throw std::runtime_error("CUDAIO::getNumChunks(): buffer size is not set");
    }
    const std::size_t fileSize = IO_Base<ItemType>::getFileSize();
    return fileSize / this->bufferSize + (fileSize % this->bufferSize != 0);
}

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

/**
 * setDeviceBufferSize()
 *
 * Precondition: newSize > 0
 *                && newSize is a multiple of sizeof(ItemType)
 * Postcondition: bufferSize is set to newSize if valid.
 */
template <class ItemType>
void CUDAIO<ItemType>::setDeviceBufferSize(std::size_t newSize) {
        if (this->devPtr != nullptr) {
            throw std::runtime_error("CUDAIO::setDeviceBufferSize(): "
                                     "cannot resize an allocated GPU buffer");
        }

        if (newSize == 0 || newSize % sizeof(ItemType) != 0) {
            throw std::invalid_argument("CUDAIO::setDeviceBufferSize(): invalid buffer size");
        }

        this->bufferSize = newSize;
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

        throw std::runtime_error(std::string("CUDAIO::open(): failed to open file: ") + std::strerror(errno));
    }

    const long itemSize = static_cast<long>(sizeof(ItemType));

    // Reader
    if (accessMode == O_RDONLY) {

        struct stat fileInfo{};

        if (fstat(this->fd, &fileInfo) == -1) {

            ::close(this->fd);

            this->fd = -1;

            throw std::runtime_error(std::string("CUDAIO::open(): fstat failed: ") + std::strerror(errno));
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

    else if (accessMode == O_WRONLY || accessMode == O_RDWR) {

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

            throw std::runtime_error(std::string("CUDAIO::open(): ftruncate failed: ") + std::strerror(errno));
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
class gpuReader : public CUDAIO<ItemType> {

  public:
    gpuReader();
    gpuReader(const std::string &fileName);
    ItemType *readChunk(std::size_t chunkIndex); // Reads a chunk of the specified size into GPU memory

    ~gpuReader() = default;
};

/* gpuReader minimal constructor
 */

template <class ItemType>
gpuReader<ItemType>::gpuReader() : CUDAIO<ItemType>() {}

/*
 * gpuReader file-based constructor
 * @param: fileName, path to the file
 * Precondition: fileName is a valid path to an existing file
 * Postcondition: file is open and ready for reading
 *
 */

template <class ItemType>
gpuReader<ItemType>::gpuReader(const std::string &fileName) : CUDAIO<ItemType>() {
    this->open(fileName, O_RDONLY | O_DIRECT);
}

/* gpuReader::readChunk
 * Reads a chunk of the file directly into GPU memory.
 * Precondition: file is open and ready for reading.
 *                  && chunk can fit into available GPU memory.
 * Postcondition: GPU memory contains the chunk content, or nullptr if the file is
 * empty.
 */

template <class ItemType>
ItemType *gpuReader<ItemType>::readChunk(std::size_t chunkIndex) {

    // Verify that the file is open.
    if (!IO_Base<ItemType>::getFileOpened()) {
        throw std::runtime_error("gpuReader::readChunk(): file not opened");
    }

    const std::size_t fileSize = static_cast<std::size_t>(this->getFileSize());

    const std::size_t bufferSize = this->getDeviceBufferSize();

    if (bufferSize == 0 || bufferSize % sizeof(ItemType) != 0) {
        throw std::runtime_error("gpuReader::readChunk(): invalid buffer size");
    }

    if (chunkIndex >= this->getNumChunks()) {
        return nullptr;
    }

    const std::size_t fileOffset = chunkIndex * bufferSize;

    const std::size_t bytesToRead = std::min(bufferSize, fileSize - fileOffset);

    IO_Base<ItemType>::setChunkSize(bytesToRead);

    // Verify that the buffer size is valid.
    if (bufferSize == 0 || bufferSize % sizeof(ItemType) != 0) {
        throw std::runtime_error("gpuReader::readChunk(): invalid buffer size");
    }

    // Check whether the requested chunk exists.
    if (chunkIndex >= this->getNumChunks()) {
        return nullptr;
    }


    // Allocate GPU memory only if no buffer exists.
    if (this->devPtr == nullptr) {

        std::size_t freeMemory = 0;
        std::size_t totalMemory = 0;

        checkResult(cudaMemGetInfo(&freeMemory, &totalMemory));

        if (bufferSize > freeMemory) {
            throw std::runtime_error("gpuReader::readChunk(): insufficient GPU memory. "
                                     "Required: " +
                                     std::to_string(bufferSize) + " bytes, available: " + std::to_string(freeMemory) +
                                     " bytes");
        }

        checkResult(cudaMalloc(&this->devPtr, bufferSize));
    }

    // Read the requested chunk into the reusable GPU buffer.
    const ssize_t readBytes = cuFileRead(this->cfHandle, this->devPtr, bytesToRead, static_cast<off_t>(fileOffset), 0);

    // Check for read errors.
    if (readBytes < 0) {
        const int savedErrno = errno;

        const std::string errorMessage = (readBytes == -1) ? std::string(std::strerror(savedErrno))
                                                           : "cuFile error code " + std::to_string(readBytes);

        throw std::runtime_error("gpuReader::readChunk(): cuFileRead failed. "
                                 "Chunk: " +
                                 std::to_string(chunkIndex) + ", Offset: " + std::to_string(fileOffset) +
                                 ", Error: " + errorMessage);
    }

    // Verify that the complete chunk was read.
    if (static_cast<std::size_t>(readBytes) != bytesToRead) {
        throw std::runtime_error("gpuReader::readChunk(): incomplete read. "
                                 "Expected: " +
                                 std::to_string(bytesToRead) + " bytes, received: " + std::to_string(readBytes) +
                                 " bytes");
    }

    return static_cast<ItemType *>(this->devPtr);
}

/*
 * gpuWriter writes data that already resides in GPU memory directly to
 * storage using NVIDIA GPUDirect Storage.
 */

template <class ItemType>
class gpuWriter : public CUDAIO<ItemType> {

  public:
    gpuWriter();

    gpuWriter(const std::string &fileName);

    // Writes numItems values directly from GPU memory to the file.
    void writeChunk(const ItemType *gpuData, std::size_t numItems, std::size_t fileOffset);

    ~gpuWriter() = default;
};

/*
 * gpuWriter minimal constructor
 *
 * Initializes an empty writer with no associated file.
 */

template <class ItemType>
gpuWriter<ItemType>::gpuWriter() : CUDAIO<ItemType>() {
    IO_Base<ItemType>::setFileSize(0);
    IO_Base<ItemType>::setNumItemsInFile(0);
}

/*
 * gpuWriter file-based constructor
 *
 */

template <class ItemType>
gpuWriter<ItemType>::gpuWriter(const std::string &fileName) : gpuWriter<ItemType>() {
    CUDAIO<ItemType>::open(fileName, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT);
}

/*
 * gpuWriter::writeChunk
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
void gpuWriter<ItemType>::writeChunk(const ItemType *gpuData, std::size_t numItems, std::size_t fileOffset) {

    if (!IO_Base<ItemType>::getFileOpened()) {
        throw std::runtime_error("gpuWriter::writeChunk(): file not opened");
    }

    if (numItems == 0) {
        return;
    }

    if (gpuData == nullptr) {
        throw std::invalid_argument("gpuWriter::writeChunk(): gpuData is nullptr");
    }

    // Ensure the file offset is aligned to the size of ItemType.
    if (fileOffset % sizeof(ItemType) != 0) {
        throw std::invalid_argument("gpuWriter::writeChunk(): unaligned item offset");
    }

    const std::size_t bytesToWrite = numItems * sizeof(ItemType);

    const std::size_t maxOffset = static_cast<std::size_t>(std::numeric_limits<off_t>::max());

    if (fileOffset > maxOffset || bytesToWrite > maxOffset - fileOffset) {
        throw std::overflow_error("gpuWriter::writeChunk(): file offset overflow");
    }

    const std::size_t requiredFileSize = fileOffset + bytesToWrite;

    // Verify that this is device memory.
    cudaPointerAttributes attributes{};

    checkResult(cudaPointerGetAttributes(&attributes, gpuData));

    if (attributes.type != cudaMemoryTypeDevice) {
        throw std::invalid_argument("gpuWriter::writeChunk(): expected CUDA device memory");
    }

    // Extend the file when necessary.
    const std::size_t currentFileSize = static_cast<std::size_t>(IO_Base<ItemType>::getFileSize());

    if (requiredFileSize > currentFileSize) {
        if (ftruncate(this->fd, static_cast<off_t>(requiredFileSize)) != 0) {
            throw std::runtime_error(std::string("gpuWriter::writeChunk(): ftruncate failed: ") + std::strerror(errno));
        }

        IO_Base<ItemType>::setFileSize(static_cast<long>(requiredFileSize));

        IO_Base<ItemType>::setNumItemsInFile(requiredFileSize / sizeof(ItemType));
    }

    // Write the chunk at its specified file offset.
    const ssize_t writtenBytes = cuFileWrite(this->cfHandle, gpuData, bytesToWrite, static_cast<off_t>(fileOffset), 0);

    if (writtenBytes < 0) {
        throw std::runtime_error("gpuWriter::writeChunk(): cuFileWrite failed with error " +
                                 std::to_string(writtenBytes));
    }

    if (static_cast<std::size_t>(writtenBytes) != bytesToWrite) {
        throw std::runtime_error("gpuWriter::writeChunk(): incomplete write");
    }
}

// Helper Utilities

inline void checkResult(cudaError_t result) {

    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(result));
    }
}

inline void checkResult(CUfileError_t result) {

    if (result.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("cuFile Error: " + std::to_string(result.err));
    }
}
#endif // CUDAIO_H
