/* ThreadsIO.h contains the ThreadsIO class that provides thread-safe I/O operations
 * supporting parallel binary I/O using system threads.
 *
 * author: Joshua Eilu for Professor Joel Adams at Calvin University.
 * date:   Summer 2026
 */

#ifndef OO_IO_INCLUDE_THREADS_IO_H
#define OO_IO_INCLUDE_THREADS_IO_H

#include "IO_Base.h"
#include <atomic>     // std::atomic
#include <cstring>    // std::memcpy
#include <fcntl.h>    // open(), O_RDONLY, O_RDWR, ...
#include <span>       // std::span
#include <sys/mman.h> // mmap(), munmap(), msync()
#include <sys/stat.h> // fstat()
#include <unistd.h>   // close(), ftruncate()


/* ThreadIOMode identifies the kind of I/O operation performed by a ThreadsIO object */
enum class ThreadIOMode { None, Read, Write };

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
    ThreadsIO(int id, int num_threads, ThreadIOMode mode);

    virtual ~ThreadsIO() = default;

    void open(const std::string &fileName) override;
    void close() override;

  protected:
    // Shared POSIX file and mapping state.
    static int sharedFd;
    static void *sharedMapBase;
    static ItemType *sharedMapData;
    static std::size_t sharedMapBytes;
    static long sharedFileSize;

    static std::atomic<ThreadIOMode> activeMode; // IO mode using the shared mapping.
    static std::atomic<bool> openFlag;           // Open-phase synchronization.
    static std::atomic<int> remainingUsers;      // Number of threads still using the mapping.

  private:
    const ThreadIOMode myMode; // mode requested by this individual object.
};

/* ThreadsIO constructor
 * @param: id, an int
 * @param: numThreads, an int
 * @param: mode, a ThreadIOMode
 * Precondition: id >= 0
 *           &&  numThreads > 0
 *           &&  mode is either ThreadIOMode::Read or ThreadIOMode::Write.
 */
template <class ItemType>
ThreadsIO<ItemType>::ThreadsIO(int id, int num_threads, ThreadIOMode mode)
    : IO_Base<ItemType>(id, num_threads), myMode(mode) {}

// Shared static member initialization
template <class ItemType>
int ThreadsIO<ItemType>::sharedFd{-1};

template <class ItemType>
void *ThreadsIO<ItemType>::sharedMapBase{nullptr};

template <class ItemType>
ItemType *ThreadsIO<ItemType>::sharedMapData{nullptr};

template <class ItemType>
std::size_t ThreadsIO<ItemType>::sharedMapBytes{0};

template <class ItemType>
long ThreadsIO<ItemType>::sharedFileSize{0};

template <class ItemType>
std::atomic<ThreadIOMode> ThreadsIO<ItemType>::activeMode{ThreadIOMode::None};

template <class ItemType>
std::atomic<bool> ThreadsIO<ItemType>::openFlag{false};

template <class ItemType>
std::atomic<int> ThreadsIO<ItemType>::remainingUsers{0};

/* open() opens and maps a file for the current read or write phase.
 * @param: fileName, a string
 * Precondition: fileName is the name of the file to be read or written.
 * Postcondition: thread 0 has opened and mapped the file for reading or writing;
 *             && the shared file information has been recorded in this object.
 */
template <class ItemType>
void ThreadsIO<ItemType>::open(const std::string &fileName) {
    if (IO_Base<ItemType>::getID() == 0) {
        ThreadIOMode currentMode = activeMode.load(std::memory_order_acquire);

        // If another phase is still active, wait until it finishes before starting a new IO
        // phase.
        while (currentMode != ThreadIOMode::None) {
            activeMode.wait(currentMode, std::memory_order_acquire);
            currentMode = activeMode.load(std::memory_order_acquire);
        }

        openFlag.store(false, std::memory_order_release);

        // Set the active mode to the requested mode and wake up other threads.
        activeMode.store(myMode, std::memory_order_release);
        activeMode.notify_all();

        if (myMode == ThreadIOMode::Read) {
            sharedFd = ::open(fileName.c_str(), O_RDONLY);

            if (sharedFd == -1) {
                perror("open");
                exit(EXIT_FAILURE);
            }

            struct stat fileInfo;

            // Get file info to retrieve file size and other details.
            if (fstat(sharedFd, &fileInfo) == -1) {
                perror("fstat");
                ::close(sharedFd);
                sharedFd = -1;
                exit(EXIT_FAILURE);
            }

            sharedFileSize = static_cast<long>(fileInfo.st_size);
        } else if (myMode == ThreadIOMode::Write) {
            sharedFileSize = IO_Base<ItemType>::getFileSize();

            sharedFd = ::open(fileName.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);

            if (sharedFd == -1) {
                perror("open");
                exit(EXIT_FAILURE);
            }

            // Resize the file to the specified size before mapping.
            if (ftruncate(sharedFd, sharedFileSize) == -1) {
                perror("ftruncate");
                ::close(sharedFd);
                sharedFd = -1;
                exit(EXIT_FAILURE);
            }
        }

        sharedMapBytes = static_cast<std::size_t>(sharedFileSize);

        if (sharedFileSize > 0) {
            int protection;
            int mappingType;

            if (myMode == ThreadIOMode::Read) {
                protection = PROT_READ;
                mappingType = MAP_PRIVATE;
            } else {
                protection = PROT_READ | PROT_WRITE;
                mappingType = MAP_SHARED;
            }

            sharedMapBase =
                mmap(nullptr, sharedMapBytes, protection, mappingType, sharedFd, 0);

            if (sharedMapBase == MAP_FAILED) {
                perror("mmap");
                ::close(sharedFd);

                sharedFd = -1;
                sharedMapBase = nullptr;

                exit(EXIT_FAILURE);
            }

            sharedMapData = reinterpret_cast<ItemType *>(sharedMapBase);
        } else {
            sharedMapBase = nullptr;
            sharedMapData = nullptr;
            sharedMapBytes = 0;
        }

        // Track the number of threads that will be using the shared mapping.
        remainingUsers.store(IO_Base<ItemType>::getNumPEs(), std::memory_order_release);

        // Wake up other threads waiting for the mapping to be ready.
        openFlag.store(true, std::memory_order_release);
        openFlag.notify_all();
    } else {
        // Wait until thread 0 starts the correct phase.
        ThreadIOMode currentMode = activeMode.load(std::memory_order_acquire);

        while (currentMode != myMode) {
            activeMode.wait(currentMode, std::memory_order_acquire);
            currentMode = activeMode.load(std::memory_order_acquire);
        }

        // Put all threads to sleep until thread 0 finishes open(), fstat(), and mmap().
        openFlag.wait(false, std::memory_order_acquire);
    }

    // Store the shared information in this individual object.
    IO_Base<ItemType>::setFileName(fileName);
    IO_Base<ItemType>::setFileSize(sharedFileSize);
    IO_Base<ItemType>::setNumItemsInFile(sharedFileSize / IO_Base<ItemType>::getItemSize());
    IO_Base<ItemType>::setFileOpened(sharedFd != -1);
}

/* close() marks this object as closed.
 * Postcondition: the final thread has closed this object, and the shared file mapping has been
 *                synchronized, unmapped, and closed.
 *             && the shared state has been reset for the next read or write phase.
 */
template <class ItemType>
void ThreadsIO<ItemType>::close() {
    // Prevent this object from closing more than once.
    if (!IO_Base<ItemType>::getFileOpened()) {
        return;
    }

    IO_Base<ItemType>::setFileOpened(false);

    // Decrease the number of threads still using the mapping.
    int oldRemaining = remainingUsers.fetch_sub(1, std::memory_order_acq_rel);

    // Only the final thread performs the shared cleanup.
    if (oldRemaining != 1) {
        return;
    }

    // A writer must save its changes before the mapping is removed.
    if (myMode == ThreadIOMode::Write && sharedMapBase != nullptr) {
        if (msync(sharedMapBase, sharedMapBytes, MS_SYNC) == -1) {
            perror("msync");
        }
    }

    if (sharedMapBase != nullptr) {
        if (munmap(sharedMapBase, sharedMapBytes) == -1) {
            perror("munmap");
        }
    }

    if (sharedFd != -1) {
        if (::close(sharedFd) == -1) {
            perror("close");
        }
    }

    // Reset the shared file information.
    sharedFd = -1;
    sharedMapBase = nullptr;
    sharedMapData = nullptr;
    sharedMapBytes = 0;
    sharedFileSize = 0;

    openFlag.store(false, std::memory_order_release);

    // The shared state is now free for the next phase.
    activeMode.store(ThreadIOMode::None, std::memory_order_release);
    activeMode.notify_all();
}

/********************************************************************
 * The ThreadReader template provides an abstraction to hide the details of
 * threads parallel input.
 *
 *
 * Note: the returned spans are valid only while the shared mapping remains
 *  alive.
 ********************************************************************/
template <class ItemType>
class ThreadReader : public ThreadsIO<ItemType> {
  public:
    ThreadReader(int id, int num_threads);

    std::span<const ItemType> readChunk();
    std::span<const ItemType> readChunkPlus(unsigned numExtras);

    ~ThreadReader() = default;
};

/* ThreadReader constructor
 * @param: id, this thread's id
 * @param: num_threads, the total number of threads
 * Precondition: id >= 0
 *            && num_threads > 0.
 * Postcondition: this reader has been initialized for threaded input.
 */
template <class ItemType>
ThreadReader<ItemType>::ThreadReader(int id, int num_threads)
    : ThreadsIO<ItemType>(id, num_threads, ThreadIOMode::Read) {}

/* readChunk()
 * Return: a std::span over the values in this thread's chunk.
 * Postcondition: this object's chunk size, first item offset, and first byte
 *                offset have been recorded.
 * Note: the returned span is a view into the mmap()ed file; it does not own
 *       the data and must not outlive the shared mapping.
 */
template <class ItemType>
std::span<const ItemType> ThreadReader<ItemType>::readChunk() {
    long numItemsInFile = IO_Base<ItemType>::getNumItemsInFile();

    if (numItemsInFile <= 0) {
        IO_Base<ItemType>::setChunkSize(0);
        IO_Base<ItemType>::setFirstItemOffset(0);
        IO_Base<ItemType>::setFirstByteOffset(0);
        return std::span<const ItemType>();
    }
    long start = 0, stop = 0;

    getChunkStartStopValues(IO_Base<ItemType>::getID(), IO_Base<ItemType>::getNumPEs(),
                            numItemsInFile, start, stop);

    long chunkSize = stop - start;
    IO_Base<ItemType>::setChunkSize(chunkSize);
    IO_Base<ItemType>::setFirstItemOffset(start);
    IO_Base<ItemType>::setFirstByteOffset(start * IO_Base<ItemType>::getItemSize());

    if (this->sharedMapData == nullptr || chunkSize <= 0) {
        return std::span<const ItemType>(); // empty view
    }

    return std::span<const ItemType>(this->sharedMapData + start, static_cast<size_t>(chunkSize));
}

/* method to read a chunk from the file (in its entirety)
 *  plus a few items from the next thread's chunk.
 *  This version is useful for search problems where the sought-after
 *  value straddles the boundary between two chunks.
 * @param: numExtras, an unsigned.
 * Precondition: numExtras is the number of additional Items to be read
 *                beyond the end of this thread's chunk.
 * Return: a std::span viewing the values of this thread's chunk
 *          plus numExtras values of the next thread's chunk
 *          for all threads except the last one.
 *
 * Note: Giving numExtras a default parameter value would let us eliminate
 *        the other readChunk() method and adhere to the DRY principle,
 *        but defining each separately lets us more clearly explain
 *        the difference between the two versions.
 */
template <class ItemType>
std::span<const ItemType> ThreadReader<ItemType>::readChunkPlus(unsigned numExtras) {
    long numItemsInFile = IO_Base<ItemType>::getNumItemsInFile();

    if (numItemsInFile <= 0) {
        IO_Base<ItemType>::setChunkSize(0);
        IO_Base<ItemType>::setFirstItemOffset(0);
        IO_Base<ItemType>::setFirstByteOffset(0);
        return std::span<const ItemType>();
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

    if (this->sharedMapData == nullptr || chunkSize <= 0) {
        return std::span<const ItemType>();
    }

    return std::span<const ItemType>(this->sharedMapData + start, static_cast<size_t>(chunkSize));
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
    ThreadWriter(int id, int num_threads, long fileSize);

    void writeChunk(const std::span<const ItemType> &values);

    ~ThreadWriter() override = default;
};


/* ThreadWriter constructor
 * @param: id, this thread's id
 * @param: num_threads, total number of threads
 * @param: fileSize, final output file size in bytes
 */
template <class ItemType>
ThreadWriter<ItemType>::ThreadWriter(int id, int num_threads, long fileSize)
    : ThreadsIO<ItemType>(id, num_threads, ThreadIOMode::Write) {
    IO_Base<ItemType>::setFileSize(fileSize);
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
void ThreadWriter<ItemType>::writeChunk(const std::span<const ItemType> &v)
{
    long totalItems = OO_IO_Base<ItemType>::getFileSize() / sizeof(ItemType);

    OO_IO_Base<ItemType>::setNumItemsInFile(totalItems);

    long start = 0, stop = 0;
    getChunkStartStopValues(
        OO_IO_Base<ItemType>::getID(),
        OO_IO_Base<ItemType>::getNumPEs(),
        totalItems,
        start,
        stop);

    long expectedChunkSize = stop - start;
    long actualChunkSize = static_cast<long>(v.size());

    if (actualChunkSize != expectedChunkSize)
    {
        fprintf(stderr,
                "Thread %d: chunk size mismatch (expected %ld, got %ld)\n",
                OO_IO_Base<ItemType>::getID(),
                expectedChunkSize,
                actualChunkSize);
        exit(EXIT_FAILURE);
    }

    OO_IO_Base<ItemType>::setChunkSize(actualChunkSize);
    OO_IO_Base<ItemType>::setFirstItemOffset(start);
    OO_IO_Base<ItemType>::setFirstByteOffset(start * sizeof(ItemType));

    if (this->sharedMapData == nullptr && actualChunkSize > 0)
    {
        fprintf(stderr, "ThreadWriter::writeChunk(): output file is not mapped\n");
        exit(EXIT_FAILURE);
    }

    size_t bytesToWrite = actualChunkSize * sizeof(ItemType);

    // Copy this thread's data into its assigned part of the mapped file.
    std::memcpy(this->sharedMapData + start, v.data(), bytesToWrite);
}
#endif // OO_IO_INCLUDE_THREADS_IO_H