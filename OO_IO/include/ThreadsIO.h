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

/* ThreadIOMode identifies the kind of I/O operation performed by a ThreadIO object */
enum class ThreadIOMode { READ, WRITE };

/********************************************************************
 * ThreadsIO is a base class for shared-memory thread backends that use
 *  POSIX threads.
 *
 * It manages the shared resources needed for thread-safe I/O operations, including file
 * descriptors and memory mappings.
 *
 * It does NOT create threads, synchronize them, or schedule chunks; *
 * It's sub-classes are ThreadReader and ThreadWriter.
 *
 ********************************************************************/
template <class ItemType>
class ThreadsIO : public OO_IO_Base<ItemType> {
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
 *           &&  fileName is the name of the file to be read or written.
 *           &&  mode is either ThreadIOMode::READ or ThreadIOMode::WRITE.
 */
template <class ItemType>
ThreadsIO<ItemType>::ThreadsIO(int id, int num_threads, ThreadIOMode mode)
    : IO_Base<ItemType>(fileName, id, num_threads), myMode(mode) {}

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

#endif // OO_IO_INCLUDE_THREADS_IO_H