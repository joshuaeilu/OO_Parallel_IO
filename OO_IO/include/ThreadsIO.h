/* ThreadsIO.h contains the ThreadsIO class that provides thread-safe I/O operations
 * supporting parallel binary I/O using system threads.
 *
 * author: Joshua Eilu for Professor Joel Adams at Calvin University.
 * date:   Summer 2026
 */

#ifndef OO_IO_INCLUDE_THREADS_IO_H
#define OO_IO_INCLUDE_THREADS_IO_H

#include "IO_Base.h"
#include <atomic>  // std::atomic
#include <cstring> // std::memcpy
#include <fcntl.h> // open(), O_RDONLY, O_RDWR, ...
#include <span>    // std::span
#include <stdexcept>
#include <sys/mman.h> // mmap(), munmap(), msync()
#include <sys/stat.h> // fstat()
#include <unistd.h>   // close(), ftruncate()
#include <vector>

/********************************************************************
 * ThreadsIO is a base class for shared-memory thread backends that use
 *  POSIX threads.
 *
 * It manages the shared resources needed for thread-safe I/O operations, including file
 * descriptors and memory mappings.
 *
 * It does NOT create or schedule threads.
 * It's sub-classes are ThreadReader and ThreadWriter.
 *
 ********************************************************************/
template <class ItemType>
class ThreadsIO : public IO_Base<ItemType> {
  public:
    ThreadsIO(int id, int numThreads);

    virtual ~ThreadsIO() = default;
};

/* ThreadsIO constructor
 * @param: id, an int
 * @param: numThreads, an int
 * Precondition: id >= 0
 *           &&  numThreads > 0
 */
template <class ItemType>
ThreadsIO<ItemType>::ThreadsIO(int id, int numThreads) : IO_Base<ItemType>(id, numThreads) {}

/********************************************************************
 * The ThreadReader template provides an abstraction to hide the details of
 * threads parallel input.
 ********************************************************************/
template <class ItemType>
class ThreadReader : public ThreadsIO<ItemType> {
  public:
    ThreadReader(int id, int numThreads);
    ThreadReader(int id, int numThreads, const std::string &fileName);

    void open(const std::string &fileName, int openMode) override;
    void close() override;

    std::vector<ItemType> readChunk();
    std::vector<ItemType> readChunkPlus(unsigned numExtras);

    ~ThreadReader() = default;

  private:
    static inline int sharedReadFd{-1};                       // file descriptor for the shared input file
    static inline void *sharedReadMapBase{nullptr};           // base address of the shared memory mapping for the input file
    static inline const ItemType *sharedReadMapData{nullptr}; // pointer to the shared memory mapping for the input file
    static inline std::size_t sharedReadMapBytes{0};          // size of the shared memory mapping for the input file
    static inline long sharedReadFileSize{0};                 // size of the shared input file
    static inline std::atomic<bool> readOpenFlag{false};      // indicates whether the shared input file is open
    static inline std::atomic<int> remainingThreadReaders{0};
};

/* ThreadReader minimal constructor
 * @param: id, this thread's id
 * @param: numThreads, the total number of threads
 * Precondition: id >= 0
 *            && numThreads > 0.
 * Postcondition: this reader has been initialized for threaded input.
 */
template <class ItemType>
ThreadReader<ItemType>::ThreadReader(int id, int numThreads) : ThreadsIO<ItemType>(id, numThreads) {}

/* ThreadReader file-based constructor
 * @param: id, this thread's id
 * @param: numThreads, the total number of threads
 * @param: fileName, the name of the file to read from
 * Precondition: id >= 0
 *            && numThreads > 0
 *            && fileName is a valid file path.
 * Postcondition: this reader has been initialized for threaded input
 *                and the file has been opened.
 */
template <class ItemType>
ThreadReader<ItemType>::ThreadReader(int id, int numThreads, const std::string &fileName) : ThreadsIO<ItemType>(id, numThreads) {
    this->open(fileName, O_RDONLY);
}

/* ThreadReader::open
 * @param: fileName, the name of the file to read from
 * @param: openMode, the mode to open the file in
 * Precondition: fileName is a valid file path
 *            && openMode == O_RDONLY.
 * Postcondition: the file has been opened for threaded input.
 */

template <class ItemType>
void ThreadReader<ItemType>::open(const std::string &fileName, int openMode) {

    if (openMode != O_RDONLY) {
        throw std::invalid_argument("ThreadReader::open(): expected O_RDONLY");
    }

    // Prevent this individual reader from opening twice.
    if (IO_Base<ItemType>::getFileOpened()) {
        return;
    }

    if (IO_Base<ItemType>::getID() == 0) {
        // The shared read mapping is not ready yet.
        readOpenFlag.store(false, std::memory_order_release);

        // Thread 0 opens the input file.
        sharedReadFd = ::open(fileName.c_str(), openMode);

        if (sharedReadFd == -1) {
            perror("ThreadReader::open");
            exit(EXIT_FAILURE);
        }

        struct stat fileInfo{};

        // Determine the size of the input file.
        if (fstat(sharedReadFd, &fileInfo) == -1) {
            perror("ThreadReader::fstat");

            ::close(sharedReadFd);
            sharedReadFd = -1;

            exit(EXIT_FAILURE);
        }

        sharedReadFileSize = static_cast<long>(fileInfo.st_size);

        sharedReadMapBytes = static_cast<std::size_t>(sharedReadFileSize);

        // mmap() cannot be used for a zero-byte file.
        if (sharedReadMapBytes > 0) {
            sharedReadMapBase = mmap(nullptr, sharedReadMapBytes, PROT_READ, MAP_PRIVATE, sharedReadFd, 0);

            if (sharedReadMapBase == MAP_FAILED) {
                perror("ThreadReader::mmap");

                ::close(sharedReadFd);

                sharedReadFd = -1;
                sharedReadMapBase = nullptr;

                exit(EXIT_FAILURE);
            }

            sharedReadMapData = static_cast<const ItemType *>(sharedReadMapBase);
        } else {
            sharedReadMapBase = nullptr;
            sharedReadMapData = nullptr;
            sharedReadMapBytes = 0;
        }

        // Every ThreadReader participating in this phase must
        // eventually call close().
        remainingThreadReaders.store(IO_Base<ItemType>::getNumPEs(), std::memory_order_release);

        // The descriptor, file size, and mapping are now ready.
        readOpenFlag.store(true, std::memory_order_release);
        readOpenFlag.notify_all();

    } else {
        // Other threads wait until thread 0 finishes opening
        // and mapping the file.
        readOpenFlag.wait(false, std::memory_order_acquire);
    }

    // Each individual ThreadReader records the shared file metadata.
    IO_Base<ItemType>::setFileName(fileName);
    IO_Base<ItemType>::setFileSize(sharedReadFileSize);
    IO_Base<ItemType>::setNumItemsInFile(sharedReadFileSize / static_cast<long>(IO_Base<ItemType>::getItemSize()));
    IO_Base<ItemType>::setFileOpened(true);
}

template <class ItemType>
void ThreadReader<ItemType>::close() {

    // This individual ThreadReader is now closed.
    IO_Base<ItemType>::setFileOpened(false);

    // Determine whether this is the final reader.
    const int previousRemaining = remainingThreadReaders.fetch_sub(1, std::memory_order_acq_rel);

    // The shared mapping remains alive until the final reader closes.
    if (previousRemaining != 1) {
        return;
    }

    // Final reader removes the shared mapping.
    if (sharedReadMapBase != nullptr) {
        if (munmap(sharedReadMapBase, sharedReadMapBytes) == -1) {
            perror("ThreadReader::munmap");
        }
    }

    // Final reader closes the shared descriptor.
    if (sharedReadFd != -1) {
        if (::close(sharedReadFd) == -1) {
            perror("ThreadReader::close");
        }
    }

    // Reset all shared reader state.
    sharedReadFd = -1;

    sharedReadMapBase = nullptr;
    sharedReadMapData = nullptr;

    sharedReadMapBytes = 0;
    sharedReadFileSize = 0;

    remainingThreadReaders.store(0, std::memory_order_release);

    // Allow the next reader phase to initialize a new mapping.
    readOpenFlag.store(false, std::memory_order_release);
    readOpenFlag.notify_all();
}

/* readChunk()
 * Return: a std::vector over the values in this thread's chunk.
 * Postcondition: this object's chunk size, first item offset, and first byte
 *                offset have been recorded.
 */
template <class ItemType>
std::vector<ItemType> ThreadReader<ItemType>::readChunk() {
    long numItemsInFile = IO_Base<ItemType>::getNumItemsInFile();

    if (numItemsInFile <= 0) {
        IO_Base<ItemType>::setChunkSize(0);
        IO_Base<ItemType>::setFirstItemOffset(0);
        IO_Base<ItemType>::setFirstByteOffset(0);
        return std::vector<ItemType>();
    }
    long start = 0, stop = 0;

    getChunkStartStopValues(IO_Base<ItemType>::getID(), IO_Base<ItemType>::getNumPEs(), numItemsInFile, start, stop);

    long chunkSize = stop - start;
    IO_Base<ItemType>::setChunkSize(chunkSize);
    IO_Base<ItemType>::setFirstItemOffset(start);
    IO_Base<ItemType>::setFirstByteOffset(start * IO_Base<ItemType>::getItemSize());

    if (this->sharedReadMapData == nullptr || chunkSize <= 0) {
        return std::vector<ItemType>(); // empty view
    }

    return std::vector<ItemType>(this->sharedReadMapData + start, this->sharedReadMapData + stop);
}

/* method to read a chunk from the file (in its entirety)
 *  plus a few items from the next thread's chunk.
 *  This version is useful for search problems where the sought-after
 *  value straddles the boundary between two chunks.
 * @param: numExtras, an unsigned.
 * Precondition: numExtras is the number of additional Items to be read
 *                beyond the end of this thread's chunk.
 * Return: a std::vector viewing the values of this thread's chunk
 *          plus numExtras values of the next thread's chunk
 *          for all threads except the last one.
 *
 * Note: Giving numExtras a default parameter value would let us eliminate
 *        the other readChunk() method and adhere to the DRY principle,
 *        but defining each separately lets us more clearly explain
 *        the difference between the two versions.
 */
template <class ItemType>
std::vector<ItemType> ThreadReader<ItemType>::readChunkPlus(unsigned numExtras) {
    long numItemsInFile = IO_Base<ItemType>::getNumItemsInFile();

    if (numItemsInFile <= 0) {
        IO_Base<ItemType>::setChunkSize(0);
        IO_Base<ItemType>::setFirstItemOffset(0);
        IO_Base<ItemType>::setFirstByteOffset(0);
        return std::vector<ItemType>();
    }

    long start = 0, stop = 0;

    int id = IO_Base<ItemType>::getID();
    int numPEs = IO_Base<ItemType>::getNumPEs();

    getChunkStartStopValues(id, numPEs, numItemsInFile, start, stop);

    if (id < numPEs - 1) {
        stop += numExtras;
    }

    if (stop > numItemsInFile) {
        stop = numItemsInFile;
    }

    long chunkSize = stop - start;

    IO_Base<ItemType>::setChunkSize(chunkSize);
    IO_Base<ItemType>::setFirstItemOffset(start);
    IO_Base<ItemType>::setFirstByteOffset(start * IO_Base<ItemType>::getItemSize());

    if (this->sharedReadMapData == nullptr || chunkSize <= 0) {
        return std::vector<ItemType>();
    }

    return std::vector<ItemType>(this->sharedReadMapData + start, this->sharedReadMapData + stop);
}

/********************************************************************
 * ThreadWriter writes binary data to a file in parallel using
 * multiple threads (shared-memory).
 *
 * Each thread writes its own chunk of data to a distinct region
 * of the file.
 *
 * It uses ThreadsIO as its superclass.
 ********************************************************************/
template <class ItemType>
class ThreadWriter : public ThreadsIO<ItemType> {
  public:
    ThreadWriter(int id, int numThreads, long fileSize);
    ThreadWriter(int id, int numThreads, long fileSize, const std::string &fileName);

    void open(const std::string &fileName, int openMode);
    void close();

    void writeChunk(const std::span<const ItemType> &values);

    ~ThreadWriter() override { close(); }

  protected:
    static inline int sharedWriteFd{-1};

    static inline void *sharedWriteMapBase{nullptr};
    static inline ItemType *sharedWriteMapData{nullptr};

    static inline std::size_t sharedWriteMapBytes{0};
    static inline long sharedWriteFileSize{0};

    static inline std::atomic<bool> writeOpenFlag{false};
    static inline std::atomic<int> remainingWriters{0};
};

/* ThreadWriter minimal constructor
 * @param: id, this thread's id
 * @param: numThreads, total number of threads
 * @param: fileSize, final output file size in bytes
 */
template <class ItemType>
ThreadWriter<ItemType>::ThreadWriter(int id, int numThreads, long fileSize) : ThreadsIO<ItemType>(id, numThreads) {
    IO_Base<ItemType>::setFileSize(fileSize);
}

/* ThreadWriter file-based constructor
 * @param: id, this thread's id
 * @param: numThreads, total number of threads
 * @param: fileSize, final output file size in bytes
 * @param: fileName, the name of the output file
 */
template <class ItemType>
ThreadWriter<ItemType>::ThreadWriter(int id, int numThreads, long fileSize, const std::string &fileName)
    : ThreadsIO<ItemType>(id, numThreads) {
    IO_Base<ItemType>::setFileSize(fileSize);
    open(fileName, O_RDWR);
}

/* ThreadWriter::open
 * @param: fileName, the name of the output file
 * @param: openMode, expected to be O_RDWR
 * Precondition: fileName is a valid file path
 *            && openMode == O_RDWR.
 * Postcondition: the output file has been opened for threaded output.
 */
template <class ItemType>
void ThreadWriter<ItemType>::open(const std::string &fileName, int openMode) {

    if ((openMode & O_ACCMODE) != O_RDWR) {
        throw std::invalid_argument("ThreadWriter::open(): expected O_RDWR access");
    }

    // Prevent this individual writer from opening twice.
    if (IO_Base<ItemType>::getFileOpened()) {
        return;
    }

    if (IO_Base<ItemType>::getID() == 0) {
        writeOpenFlag.store(false, std::memory_order_release);

        // The desired output size was supplied to the constructor.
        sharedWriteFileSize = IO_Base<ItemType>::getFileSize();

        if (sharedWriteFileSize < 0) {
            throw std::invalid_argument("ThreadWriter::open(): file size must be nonnegative");
        }

        // Thread 0 creates/truncates the output file.
        sharedWriteFd = ::open(fileName.c_str(), openMode | O_CREAT | O_TRUNC, 0644);

        if (sharedWriteFd == -1) {
            perror("ThreadWriter::open");
            exit(EXIT_FAILURE);
        }

        // Resize the file before mmap().
        if (ftruncate(sharedWriteFd, sharedWriteFileSize) == -1) {

            perror("ThreadWriter::ftruncate");

            ::close(sharedWriteFd);
            sharedWriteFd = -1;

            exit(EXIT_FAILURE);
        }

        sharedWriteMapBytes = static_cast<std::size_t>(sharedWriteFileSize);

        // mmap() cannot map a zero-byte file.
        if (sharedWriteMapBytes > 0) {
            sharedWriteMapBase = mmap(nullptr, sharedWriteMapBytes, PROT_READ | PROT_WRITE, MAP_SHARED, sharedWriteFd, 0);

            if (sharedWriteMapBase == MAP_FAILED) {
                perror("ThreadWriter::mmap");

                ::close(sharedWriteFd);

                sharedWriteFd = -1;
                sharedWriteMapBase = nullptr;

                exit(EXIT_FAILURE);
            }

            sharedWriteMapData = static_cast<ItemType *>(sharedWriteMapBase);
        } else {
            sharedWriteMapBase = nullptr;
            sharedWriteMapData = nullptr;
            sharedWriteMapBytes = 0;
        }

        remainingWriters.store(IO_Base<ItemType>::getNumPEs(), std::memory_order_release);

        // Shared output mapping is ready for all writers.
        writeOpenFlag.store(true, std::memory_order_release);

        writeOpenFlag.notify_all();

    } else {
        // Other threads wait for thread 0 to finish
        // open(), ftruncate(), and mmap().
        writeOpenFlag.wait(false, std::memory_order_acquire);
    }

    IO_Base<ItemType>::setFileName(fileName);
    IO_Base<ItemType>::setFileSize(sharedWriteFileSize);

    IO_Base<ItemType>::setNumItemsInFile(sharedWriteFileSize / static_cast<long>(IO_Base<ItemType>::getItemSize()));

    IO_Base<ItemType>::setFileOpened(true);
}

/* ThreadWriter::close
 * Precondition: the output file has been opened for threaded output.
 * Postcondition: the output file has been closed for threaded output.
 */
template <class ItemType>
void ThreadWriter<ItemType>::close() {
    // Prevent this individual writer from closing twice.
    if (!IO_Base<ItemType>::getFileOpened()) {
        return;
    }

    IO_Base<ItemType>::setFileOpened(false);

    // Determine whether this is the final writer.
    const int previousRemaining = remainingWriters.fetch_sub(1, std::memory_order_acq_rel);

    // Shared resources remain alive until the final writer.
    if (previousRemaining != 1) {
        return;
    }

    // Ensure changes to the shared mapping are written to disk.
    if (sharedWriteMapBase != nullptr) {
        if (msync(sharedWriteMapBase, sharedWriteMapBytes, MS_SYNC) == -1) {

            perror("ThreadWriter::msync");
        }
    }

    // Remove the shared mapping.
    if (sharedWriteMapBase != nullptr) {
        if (munmap(sharedWriteMapBase, sharedWriteMapBytes) == -1) {

            perror("ThreadWriter::munmap");
        }
    }

    // Close the shared output file descriptor.
    if (sharedWriteFd != -1) {
        if (::close(sharedWriteFd) == -1) {
            perror("ThreadWriter::close");
        }
    }

    // Reset writer-specific shared state.
    sharedWriteFd = -1;

    sharedWriteMapBase = nullptr;
    sharedWriteMapData = nullptr;

    sharedWriteMapBytes = 0;
    sharedWriteFileSize = 0;

    remainingWriters.store(0, std::memory_order_release);

    writeOpenFlag.store(false, std::memory_order_release);

    writeOpenFlag.notify_all();
}

/* writeChunk() writes this thread's chunk to the mapped output file.
 * @param: v, a std::span containing this thread's Items.
 * Precondition: v contains exactly the Items assigned to this thread
 *            && the output file has been mapped into memory
 *            && all threads write to non-overlapping chunks.
 * Postcondition: v's values have been copied into this thread's assigned
 *                portion of the mapped output file.
 */
template <class ItemType>
void ThreadWriter<ItemType>::writeChunk(const std::span<const ItemType> &v) {
    long totalItems = IO_Base<ItemType>::getFileSize() / sizeof(ItemType);

    IO_Base<ItemType>::setNumItemsInFile(totalItems);

    long start = 0, stop = 0;
    getChunkStartStopValues(IO_Base<ItemType>::getID(), IO_Base<ItemType>::getNumPEs(), totalItems, start, stop);

    long expectedChunkSize = stop - start;
    long actualChunkSize = static_cast<long>(v.size());

    if (actualChunkSize != expectedChunkSize) {
        fprintf(stderr, "Thread %d: chunk size mismatch (expected %ld, got %ld)\n", IO_Base<ItemType>::getID(), expectedChunkSize,
                actualChunkSize);
        exit(EXIT_FAILURE);
    }

    IO_Base<ItemType>::setChunkSize(actualChunkSize);
    IO_Base<ItemType>::setFirstItemOffset(start);
    IO_Base<ItemType>::setFirstByteOffset(start * sizeof(ItemType));

    if (this->sharedWriteMapData == nullptr && actualChunkSize > 0) {
        fprintf(stderr, "ThreadWriter::writeChunk(): output file is not mapped\n");
        exit(EXIT_FAILURE);
    }

    size_t bytesToWrite = actualChunkSize * sizeof(ItemType);

    // Copy this thread's data into its assigned part of the mapped file.
    std::memcpy(this->sharedWriteMapData + start, v.data(), bytesToWrite);
}
#endif // OO_IO_INCLUDE_THREADS_IO_H