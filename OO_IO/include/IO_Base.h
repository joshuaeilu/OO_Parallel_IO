/* IO_Base.h contains the base class that supports different parallel I/O backends for reading and
 * writing binary files.
 *  @author: Joshua Eilu for Professor Joel Adams at Calvin University.
 *  @date:   Summer 2026
 */

#ifndef IO_BASE_H
#define IO_BASE_H

#include <chrono>  // std::chrono
#include <cstdio>  // fprintf(), printf(), stderr
#include <cstdlib> // exit(), EXIT_FAILURE
#include <string>  // std::string
#include <vector>  // std::vector
#include <climits> // INT_MAX
#include <cstddef> // std::size_t
#include <iostream>
/* IO_Base is a templated base class that supports parallel binary I/O
 * using MPI-IO (processes) or POSIX I/O (threads).
 * It stores information about the file and each PE’s chunk, and provides
 * helper functions for derived classes.
 *
 * It's subclasses are MPIProcessesIO (MPI), ThreadsIO (POSIX threads) and CUDAIO (CUDA).
 */

template <class ItemType> class IO_Base {
  public:
    IO_Base(int id, int numPEs);
    virtual ~IO_Base() = default;

    virtual void open(const std::string &fileName, int fileOpenMode) = 0;  // Open and close the corresponding
    virtual void close() = 0; // file for reading or writing.

    int getID()                    const { return myID; }
    int getNumPEs()                const { return myNumPEs; }
    long getItemSize()             const { return myItemSize; }
    std::string getFileName()      const { return myFileName; }
    long getNumItemsInFile()       const { return myNumItemsInFile; }
    long getChunkSize()            const { return myChunkSize; }
    long getFirstItemOffset()      const { return myFirstItemOffset; }
    long getFirstByteOffset()      const { return myFirstByteOffset; }
    long getFileSize()             const { return myFileSize; }
    bool getFileOpened()           const { return didFileOpen; }

  protected:
    void setID(int newID);
    void setNumPEs(int newNumPEs);
    void setNumItemsInFile(long numItemsInFile)     { myNumItemsInFile = numItemsInFile; }
    void setFileSize(long fileSize)                 { myFileSize = fileSize; }
    void setChunkSize(long chunkSize)               { myChunkSize = chunkSize; }
    void setFirstItemOffset(long firstItemOffset)   { myFirstItemOffset = firstItemOffset; }
    void setFirstByteOffset(long firstByteOffset)   { myFirstByteOffset = firstByteOffset; }
    void setFileOpened(bool opened)                 { didFileOpen = opened; }
    void setFileName(const std::string &fileName)   { myFileName = fileName; }

  private:
    int myID;               // this PE's thread id or MPI rank
    int myNumPEs;           // number of threads or MPI processes
    int myItemSize;         // size of 1 Item, in bytes
    std::string myFileName; // file being read/written
    bool didFileOpen;       // true iff a file was opened successfully

    // these attributes are unknown until read or write is called
    long myNumItemsInFile;  // total Items in the file
    long myFileSize;        // size of file in bytes
    long myChunkSize;       // this PE's chunk size, in Items
    long myFirstItemOffset; // this PE's first chunk offset (Item #)
    long myFirstByteOffset; // this PE's first chunk offset (byte #)
};

/* IO_Base() constructor initializes the instance variables of the IO_Base class.
 * @param: id, an integer
 * @param: numPEs, an integer
 * Precondition: id >= 0 && numPEs > 0.
 * Precondition: fileName is the name of a file containing
 *                binary-format values of type ItemType
 *           &&  id is a thread id or MPI process
 *           &&  numPEs is the number of threads or processes.
 * Postcondition: every instance variable has been initialized;
 *           &&  the per-chunk fields are set to defaults until a read/write.
 */
template <class ItemType>
IO_Base<ItemType>::IO_Base(int id, int numPEs)
    : myItemSize(sizeof(ItemType)), myFileName(""), didFileOpen(false),
      myNumItemsInFile(0), myFileSize(0), myChunkSize(0), myFirstItemOffset(0), myFirstByteOffset(0) {
    if (id >= 0) {
        myID = id;
    } else {
        fprintf(stderr, "\nIO_Base(): id must be non-negative\n\n");
        exit(1);
    }
    if (numPEs > 0) {
        myNumPEs = numPEs;
    } else {
        fprintf(stderr, "\nIO_Base(): numPEs must be positive\n\n");
        exit(1);
    }
}

/* setID() is a parameter-checking setter method for the id instance variable.
 * @param: newID, an integer
 * Precondition: newID >= 0.
 * Postcondition: myID has been set to newID.
 */
template <class ItemType> void IO_Base<ItemType>::setID(int newID) {
    if (newID < 0) {
        fprintf(stderr, "\nIO_Base::setID(): bad id (%d)\n\n", newID);
        exit(1);
    }
    myID = newID;
}

/* setNumPEs() is a parameter-checking setter method for the numPEs instance variable.
 * @param: newNumPEs, an integer
 * Precondition: newNumPEs > 0.
 * Postcondition: myNumPEs has been set to newNumPEs.
 */
template <class ItemType> void IO_Base<ItemType>::setNumPEs(int newNumPEs) {
    if (newNumPEs <= 0) {
        fprintf(stderr, "\nIO_Base::setNumPEs(): bad numPEs (%d)\n\n", newNumPEs);
        exit(1);
    }
    myNumPEs = newNumPEs;
}

/* Calculate the start and stop values for this PE's
 *  contiguous chunk of a set of loop-iterations, 0..REPS-1,
 *  so that PEs' chunk-sizes are equal (or nearly so).
 *
 * @param: id, an int containing this PE's id (thread id or MPI rank)
 * @param: numPEs, an int containing the number of PEs
 * @param: REPS, a const unsigned containing the for loop's iteration total
 * Precondition: id == this thread's id or MPI process's rank
 *            && numPEs == the number of threads or MPI processes
 *            && REPS == the total number of 0-based loop iterations needed
 *            && numPEs <= REPS
 *            && REPS < 2^32
 * @param: start, a long reference through which the
 *          starting value of this PE's chunk should be returned
 * @param: stop, a long reference through which the
 *          stopping value of this PE's chunk should be returned
 * Postcondition: start == this PE's first iteration value
 *             && stop == this PE's last iteration value + 1.
 */
inline void getChunkStartStopValues(int id, int numPEs, const long REPS, long &start, long &stop) {
    // check precondition before proceeding
    if (numPEs > REPS) {
        if (id == 0) {
            printf("\n*** Number of PEs (%d) exceeds REPS (%ld)\n", numPEs, REPS);
            printf("*** Please run using PEs less than or equal to %ld\n\n", REPS);
        }
        exit(EXIT_FAILURE);
        return;
    }

    // compute the chunk size that works in many cases
    long chunkSize1 = (REPS + numPEs - 1) / numPEs; // integer ceiling
    long begin = id * chunkSize1;
    long end = begin + chunkSize1;
    // see if there are any leftover iterations
    long remainder = REPS % numPEs;
    // If remainder == 0, chunkSize1 = chunk-size for all PEs;
    // If remainder != 0, chunkSize1 = chunk-size for p_0..p_remainder-1
    //   but for PEs p_remainder..p_numPEs-1
    //   recompute begin and end using a smaller-by-1 chunk size, chunkSize2.
    if (remainder > 0 && id >= remainder) {
        long chunkSize2 = chunkSize1 - 1;
        long remainderBase = remainder * chunkSize1;
        long peOffset = (id - remainder) * chunkSize2;
        begin = remainderBase + peOffset;
        end = begin + chunkSize2;
    }
    // pass back this PE's begin and end values via start and stop
    start = begin;
    stop = end;
}

/* Timer class
 * Purpose: Provides a simple stopwatch for measuring elapsed wall-clock time.
 *          The timer may be started and stopped multiple times, accumulating
 *          the elapsed time across each interval until reset().
 */
class Timer {
  private:
    bool running;
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::chrono::duration<double> accumulated_time;

  public:
    // Constructor initializes the timer's value to zero
    Timer() : running(false), accumulated_time(std::chrono::duration<double>::zero()) {}

    // Starts the timer
    void start() {
        if (!running) {
            start_time = std::chrono::steady_clock::now();
            running = true;
        }
    }

    // Stops the timer but does not reset its value, accumulating the elapsed time
    void stop() {
        if (running) {
            auto end_time = std::chrono::steady_clock::now();
            accumulated_time += std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
            running = false;
        }
    }

    // Resets the timer's value to zero only if it is stopped
    void reset() {
        if (!running) {
            accumulated_time = std::chrono::duration<double>::zero();
        }
    }

    // Returns the current value as a double (in seconds).
    // Note: Most useful when stopped to get the exact accumulated interval.
    double getTime() const {
        if (running) {
            // If called while running, it dynamically calculates the time
            // accumulated so far plus the current active lap.
            auto current_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> current_lap = current_time - start_time;
            return (accumulated_time + current_lap).count();
        }
        return accumulated_time.count();
    }
};

#endif