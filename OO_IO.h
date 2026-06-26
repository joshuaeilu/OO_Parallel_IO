/*   OO_IO.h (Version 3) declares C++ templates that use MPI and OPENMP to read
 *     and write binary files.
 *   - MPIProcessReader and MPIProcessWriter uses Processes or Cores for
 *     execution
 *   - ThreadReader and ThreadWriter uses Threads for execution.
 *
 *   The template allows you to pass a type-parameter indicating the type of the
 *   data in the file.
 *
 *   Note: Divides the MPI and OpenMP logic so that no necessary work is needed
 *   by the user to use either one of them.
 *   - MPI_Dataype no longer needs to be passed by the user (automatically
 * derived from the type parameter)
 *
 *   @author: Joel C. Adams, for CS 374 at Calvin University.
 *   @date:   Summer 2026
 *
 *   Version History:
 *   Version 1 (pure MPI), Fall 2023, assumes that the program
 *      using OO_IO calls MPI_Init() and MPI_Finalize().
 *   Version 2 (for OpenMP), Spring 2025, modifies
 *      - the constructors to (if necessary) call MPI_Init_thread()
 *      - the destructor to (if necessary) call MPI_Finalize()
 *        so that a multithreaded program need not call them explicitly.
 *      Also adds method MPIProcessReader::getChunkPlus(int extras) for
 *        reading a chunk plus extras items from the next PE's chunk
 *        (useful for search problems where the target spans chunk boundaries).
 *   Version 2.1, Spring 2026 fixes a missing #include for GNU gcc and
 *        adds some commented-out hints for using the library with Pthreads.
 *
 * Note: OO_IO uses 'PEs' (processing elements) as a synonym
 *        for threads or processes.
 *
 */

#ifndef OO_IO
#define OO_IO

#include <climits>    // INT_MAX
#include <cstdio>     // fprintf(), perror(), stderr
#include <cstdlib>    // exit(), EXIT_FAILURE
#include <fcntl.h>    // open(), O_RDONLY
#include <mpi.h>      // C MPI
#include <omp.h>      // OpenMP
#include <string>     // std::string
#include <vector>     // std::vector
#include <unistd.h>   // close()
#include <sys/stat.h> // fstat()
#include <chrono>

/* Helper function declarations.
 * The definitions appear later in this file.
 * 'inline' prevents multiple-definition errors when included in multiple files.
 */
template <typename T>
inline MPI_Datatype mpiType();
inline void checkResult(int result);
inline void getChunkStartStopValues(int id, int numPEs, const unsigned REPS,
                                    long &start, long &stop);

/********************************************************************
 * OO_IO_Base is a templated base class that supports parallel binary I/O
 * using MPI-IO (processes) or POSIX I/O (threads).
 * It stores information about the file and each PE’s chunk,
 * and provides helper functions for derived classes.
 *
 * Its subclasses are ProcessesIO (MPI) and ThreadsIO (POSIX threads).
 ********************************************************************/

template <class ItemType>
class OO_IO_Base
{
public:
    OO_IO_Base(const std::string &fileName, int id, int numPEs,
               MPI_Datatype mpiType = MPI_DATATYPE_NULL);
    virtual ~OO_IO_Base();

    int getID() const { return myID; }
    int getNumPEs() const { return myNumPEs; }
    long getItemSize() const { return myItemSize; }
    std::string getFileName() const { return myFileName; }
    MPI_File &getFileHandle() { return myFileHandle; }
    MPI_Datatype getMPIType() const { return myMPIType; }
    long getNumItemsInFile() const { return myNumItemsInFile; }
    long getChunkSize() const { return myChunkSize; }
    long getFirstItemOffset() const { return myFirstItemOffset; }
    long getFirstByteOffset() const { return myFirstByteOffset; }
    long getFileSize() const { return myFileSize; }
    bool getFileOpened() const { return didFileOpen; }
    bool getUsesMPI() const { return usesMPI; }

protected:
    void setID(int newID);
    void setNumPEs(int newNumPEs);
    void setNumItemsInFile(long numItemsInFile)
    {
        myNumItemsInFile = numItemsInFile;
    }
    void setFileSize(MPI_Offset fileSize) { myFileSize = fileSize; }
    void setChunkSize(long chunkSize) { myChunkSize = chunkSize; }
    void setFirstItemOffset(long firstItemOffset)
    {
        myFirstItemOffset = firstItemOffset;
    }
    void setFirstByteOffset(long firstByteOffset)
    {
        myFirstByteOffset = firstByteOffset;
    }
    void setFileOpened(bool opened) { didFileOpen = opened; }
    void setUsesMPI(bool value) { usesMPI = value; }

private:
    int myID;               // this PE's thread id or MPI rank
    int myNumPEs;           // number of threads or MPI processes
    int myItemSize;         // size of 1 Item, in bytes
    std::string myFileName; // file being read/written
    MPI_Datatype myMPIType; // the MPI equivalent of ItemType
    MPI_File myFileHandle;  // MPI handle (MPI backend only)
    bool didFileOpen;       // true iff a file was opened successfully
    bool usesMPI;           // true iff this PE uses MPI-IO

    // these attributes are unknown until read or write is called
    long myNumItemsInFile;  // total Items in the file
    MPI_Offset myFileSize;  // size of file in bytes
    long myChunkSize;       // this PE's chunk size, in Items
    long myFirstItemOffset; // this PE's chunk offset (Item #)
    long myFirstByteOffset; // this PE's chunk offset (byte #)
};

/* OO_IO_Base constructor
 * @param: fileName, a string
 * @param: id, an int
 * @param: numPEs, an int
 * @param: mpiType, the MPI_Datatype
 * Precondition: id >= 0 && numPEs > 0.
 * Precondition: fileName is the name of a file containing
 *                binary-format values of type ItemType
 *           &&  id is a thread id or MPI process
 *           &&  numPEs is the number of threads or prrankocesses.
 *           &&  mpiType is the MPI equivalent of ItemType
 * Postcondition: every instance variable has been initialized;
 *           &&  the per-chunk fields are set to defaults until a read/write.
 */
template <class ItemType>
OO_IO_Base<ItemType>::OO_IO_Base(const std::string &fileName, int id,
                                 int numPEs, MPI_Datatype mpiType)
{
    myFileName = fileName;
    myMPIType = mpiType;
    myItemSize = sizeof(ItemType);
    usesMPI = false;
    if (id >= 0)
    {
        myID = id;
    }
    else
    {
        fprintf(stderr, "\nOO_IO_Base(): id must be non-negative\n\n");
        exit(1);
    }
    if (numPEs > 0)
    {
        myNumPEs = numPEs;
    }
    else
    {
        fprintf(stderr, "\nOO_IO_Base(): numPEs must be positive\n\n");
        exit(1);
    }

    // set other attributes to defaults for now
    myNumItemsInFile = 0;
    myFileSize = 0;
    myChunkSize = 0;
    myFirstItemOffset = 0;
    myFirstByteOffset = 0;
    didFileOpen = false;
}

/* parameter-checking setter methods for id, numPEs
 */
template <class ItemType>
void OO_IO_Base<ItemType>::setID(int newID)
{
    if (newID < 0)
    {
        fprintf(stderr, "\nOO_IO_Base::setID(): bad id (%d)\n\n", newID);
        exit(1);
    }
    myID = newID;
}

template <class ItemType>
void OO_IO_Base<ItemType>::setNumPEs(int newNumPEs)
{
    if (newNumPEs < 0)
    {
        fprintf(stderr, "\nOO_IO_Base::setNumPEs(): bad numPEs (%d)\n\n",
                newNumPEs);
        exit(1);
    }
    myNumPEs = newNumPEs;
}

/* OO_IO_Base destructor: closes the shared MPI file at end-of-life.
 * Postcondition: if this object opened an MPI file and MPI has not yet
 *                been finalized, that file handle has been closed.
 * Notes: - The thread backend manages its own POSIX descriptor in
 *          ~ThreadsIO(), so usesMPI gates the MPI close here to prevent
 *          errors when a ThreadReader is used i.e. (MPI is not initialized).
 */
template <class ItemType>
OO_IO_Base<ItemType>::~OO_IO_Base()
{
    // Only attempt close if we successfully opened the file earlier and we are
    // using MPI-IO.
    if (didFileOpen && usesMPI)
    {
        int finalized = 0;
        MPI_Finalized(&finalized); // ask MPI whether Finalize has already run
        if (!finalized)
        {
            // Safe to close the MPI file handle while MPI is still active.
            int result = MPI_File_close(&myFileHandle);
            checkResult(result);
        }
    }
}

/* ----------------------------------------------------------------------
 * MPI_Runtime: a single, process-wide MPI initializer/finalizer.
 *
 * MPI_Init() and MPI_Finalize() must each be called exactly once per
 *  process.  A function-local static object (see mpiRuntime() below) is
 *  guaranteed by C++ to be constructed exactly once, which lets this
 *  library manage MPI's lifetime without the user calling MPI_Init()
 *  or MPI_Finalize() explicitly.
 * -------------------------------------------------------------------- */
class MPI_Runtime
{
public:
    MPI_Runtime()
    {
        int initialized = 0;
        MPI_Initialized(&initialized);

        if (!initialized)
        {
            // MPI Initialization
            MPI_Init(nullptr, nullptr);
            initializedMPI = true;
        }

        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        MPI_Comm_size(MPI_COMM_WORLD, &numPEs);
    }

    ~MPI_Runtime()
    {
        int finalized = 0;
        MPI_Finalized(&finalized);

        // Only finalize MPI if:
        // 1. We initialized it, AND
        // 2. It has not already been finalized.
        if (initializedMPI && !finalized)
        {
            // MPI Finalization
            MPI_Finalize();
        }
    }

    int getRank() const { return myRank; }
    int getNumPEs() const { return numPEs; }

private:
    bool initializedMPI = false; // tracks whether this object initialized MPI.
    int myRank = 0;              // this process's MPI rank.
    int numPEs = 1;              // total number of MPI processes.
};

/*
 *   Ensure a single process-wide MPI runtime (init/finalize).
 *   Call mpiRuntime() before any MPI calls.
 */
inline MPI_Runtime &mpiRuntime()
{
    static MPI_Runtime instance;
    return instance;
}

/********************************************************************
 * ProcessesIO opens a file for parallel MPI-IO operations.
 *
 * It's subclasses are MPIProcessReader and MPIProcessWriter.
 ********************************************************************/

template <class ItemType>
class ProcessesIO : public OO_IO_Base<ItemType>
{
public:
    ProcessesIO(const std::string &fileName, int mpiMode);
    virtual ~ProcessesIO() = default;
};

/* ProcessesIO constructor
 * @param: fileName, a string
 * @param: mpiMode, an int
 * Precondition : fileName is the name of a file containing binary-format values
 *                 of type ItemType
 *          && mpiMode is a valid MPI file-open mode e.g. MPI_MODE_RDONLY
 * Postcondition: MPI has been initialized (via mpiRuntime()), the file
 *                has been opened collectively for parallel I/O, and this
 *                rank's id and numPEs have been recorded.
 * Note: ItemType's MPI_Datatype is obtained from mpiType<ItemType>();
 *       this rank and the process count come from mpiRuntime(), so the
 *       caller does not pass them.
 */
template <class ItemType>
ProcessesIO<ItemType>::ProcessesIO(const std::string &fileName, int mpiMode)
    : OO_IO_Base<ItemType>(fileName, mpiRuntime().getRank(),
                           mpiRuntime().getNumPEs(), mpiType<ItemType>())
{

    OO_IO_Base<ItemType>::setUsesMPI(true);

    // Open the file for parallel input using MPI-IO
    int result = MPI_File_open(MPI_COMM_WORLD, fileName.c_str(), mpiMode,
                               MPI_INFO_NULL, &this->getFileHandle());
    checkResult(result);
    OO_IO_Base<ItemType>::setFileOpened(true);
}

/********************************************************************
 * The MPIProcessReader template provides an abstraction to hide the details of
 * MPI-IO parallel input.
 *
 * It uses ProcessesIO as its superclass.
 ********************************************************************/
template <class ItemType>
class MPIProcessReader
    : public ProcessesIO<ItemType>
{
public:
    MPIProcessReader(const std::string &fileName);
    std::vector<ItemType> readChunk();
    std::vector<ItemType> readChunkPlus(unsigned numExtras);
    virtual ~MPIProcessReader() = default;
};

/* MPIProcessReader constructor
 * @param: fileName, a string
 * Precondition: fileName is the name of a file containing
 * Postcondition: the file has been opened for parallel input
 *           &&  each instance variable have been initialized
 *                as appropriate for this PE using the file's info.
 */
template <class ItemType>
MPIProcessReader<ItemType>::MPIProcessReader(const std::string &fileName)
    : ProcessesIO<ItemType>(fileName, MPI_MODE_RDONLY) {}

/* method to read a chunk from the file (in its entirety).
 * Return: a vector containing the values of this PE's chunk.
 * Note: vector was chosen as the return-type because it
 *        uses contiguous memory and provides a move-constructor.
 */
template <class ItemType>
std::vector<ItemType> MPIProcessReader<ItemType>::readChunk()
{
    // Note: We could compute the following attributes in the constructor,
    //  but do them here for symmetry with MPIProcessWriter
    MPI_Offset fileSize;
    MPI_File_get_size(OO_IO_Base<ItemType>::getFileHandle(), &fileSize);
    OO_IO_Base<ItemType>::setFileSize(fileSize);
    // Note: EOF char seems inconsistent on different platforms;
    //  if char tests fail and off-by-one, uncomment the next 3 lines
    //   if (std::is_same<ItemType, char>::value) {         // if ItemType is
    //   char
    //      --fileSize;                                     // ignore EOF char
    //   }
    OO_IO_Base<
        ItemType>::setNumItemsInFile(fileSize /
                                     OO_IO_Base<ItemType>::getItemSize());

    long start = 0, stop = 0;
    getChunkStartStopValues(OO_IO_Base<ItemType>::getID(),
                            OO_IO_Base<ItemType>::getNumPEs(),
                            OO_IO_Base<ItemType>::getNumItemsInFile(),
                            start, stop);
    OO_IO_Base<ItemType>::setChunkSize(stop - start);
    OO_IO_Base<ItemType>::setFirstItemOffset(start);
    OO_IO_Base<
        ItemType>::setFirstByteOffset(start *
                                      OO_IO_Base<ItemType>::getItemSize());

    MPI_Status status;
    unsigned long itemsRead = 0;
    unsigned long itemsToRead = OO_IO_Base<ItemType>::getChunkSize();
    std::vector<ItemType> v(itemsToRead);
    int readResult = 0;
    // handle very large files where chunkSize > INT_MAX
    while (itemsToRead > INT_MAX)
    {
        readResult =
            MPI_File_read_at(OO_IO_Base<ItemType>::getFileHandle(),
                             OO_IO_Base<ItemType>::getFirstByteOffset() +
                                 itemsRead,
                             v.data() + itemsRead, INT_MAX,
                             OO_IO_Base<ItemType>::getMPIType(), &status);
        checkResult(readResult);
        itemsRead += INT_MAX;
        itemsToRead -= INT_MAX;
    }
    // read in remaining Items (or if itemsToRead <= INT_MAX initially)
    readResult =
        MPI_File_read_at(OO_IO_Base<ItemType>::getFileHandle(),
                         OO_IO_Base<ItemType>::getFirstByteOffset() +
                             itemsRead,
                         v.data() + itemsRead, itemsToRead,
                         OO_IO_Base<ItemType>::getMPIType(), &status);
    checkResult(readResult);

    return v;
}

/* method to read a chunk from the file (in its entirety)
 *  plus a few items from the next PE's chunk.
 *  This version of the method is useful for search problems where
 *  the sought-after value straddles the boundary between two chunks.
 * @param: numExtras, an unsigned.
 * Precondition: numExtras is the number of additional Items to be read
 *                beyond the end of this PE's chunk.
 * Return: a vector containing the values of this PE's chunk
 *          plus numExtras values of the next PE's chunk
 *          for all PEs except the last one.
 *
 * Note: Giving numExtras a default parameter value would let us eliminate
 *        the other readChunk() method and adhere to the DRY principle,
 *        but defining each separately lets us more clearly explain
 *        the difference between the two versions.
 */
template <class ItemType>
std::vector<ItemType>
MPIProcessReader<ItemType>::readChunkPlus(unsigned numExtras)
{
    // Note: We could compute the following attributes in the constructor,
    //  but do them here for symmetry with MPIProcessWriter
    MPI_Offset fileSize;
    MPI_File_get_size(OO_IO_Base<ItemType>::getFileHandle(), &fileSize);
    OO_IO_Base<ItemType>::setFileSize(fileSize);
    // Note: EOF char seems inconsistent on different platforms;
    //  if char tests fail and off-by-one, uncomment the next 3 lines
    //   if (std::is_same<ItemType, char>::value) {         // if ItemType is
    //   char
    //      --fileSize;                                     // ignore EOF char
    //   }
    OO_IO_Base<
        ItemType>::setNumItemsInFile(fileSize /
                                     OO_IO_Base<ItemType>::getItemSize());

    long numItemsInFile = OO_IO_Base<ItemType>::getNumItemsInFile();
    long start = 0, stop = 0;
    int id = OO_IO_Base<ItemType>::getID();
    int numPEs = OO_IO_Base<ItemType>::getNumPEs();
    getChunkStartStopValues(id, numPEs, numItemsInFile, start, stop);
    if (id < numPEs - 1)
    {
        stop += numExtras;
    }
    if (stop > numItemsInFile)
    {
        stop = numItemsInFile;
    }
    OO_IO_Base<ItemType>::setChunkSize(stop - start);
    OO_IO_Base<ItemType>::setFirstItemOffset(start);
    OO_IO_Base<
        ItemType>::setFirstByteOffset(start *
                                      OO_IO_Base<ItemType>::getItemSize());

    MPI_Status status;
    unsigned long itemsRead = 0;
    unsigned long itemsToRead = OO_IO_Base<ItemType>::getChunkSize();
    std::vector<ItemType> v(itemsToRead);
    int readResult = 0;
    // handle very large files where chunkSize > INT_MAX
    while (itemsToRead > INT_MAX)
    {
        readResult =
            MPI_File_read_at(OO_IO_Base<ItemType>::getFileHandle(),
                             OO_IO_Base<ItemType>::getFirstByteOffset() +
                                 itemsRead,
                             v.data() + itemsRead, INT_MAX,
                             OO_IO_Base<ItemType>::getMPIType(), &status);
        checkResult(readResult);
        itemsRead += INT_MAX;
        itemsToRead -= INT_MAX;
    }
    // read in remaining Items (or if itemsToRead <= INT_MAX initially)
    readResult =
        MPI_File_read_at(OO_IO_Base<ItemType>::getFileHandle(),
                         OO_IO_Base<ItemType>::getFirstByteOffset() +
                             itemsRead,
                         v.data() + itemsRead, itemsToRead,
                         OO_IO_Base<ItemType>::getMPIType(), &status);
    checkResult(readResult);

    return v;
}

/*******************************************************************
 * The MPIProcessWriter template provides an abstraction to hide the
 *  details of MPI-IO parallel output.
 *
 * It uses ProcessesIO as its superclass.
 ******************************************************************/

template <class ItemType>
class MPIProcessWriter
    : public ProcessesIO<ItemType>
{
public:
    MPIProcessWriter(const std::string &fileName);
    void writeChunk(const std::vector<ItemType> &v);

    virtual ~MPIProcessWriter() = default;
};

/* MPIProcessWriter constructor
 * @param: fileName, a string
 * Precondition: fileName is the name of an output file
 *                to which binary-format values
 *                of type ItemType are to be written.
 * Postcondition: the file has been opened for parallel output
 *             &&  each instance variable have been initialized
 *                  as appropriate for this PE using id, numPEs,
 *                  and size info from the file.
 */
template <class ItemType>
MPIProcessWriter<ItemType>::MPIProcessWriter(const std::string &fileName)
    : ProcessesIO<ItemType>(fileName, MPI_MODE_CREATE | MPI_MODE_WRONLY) {}

/* method to write this PE's chunk to the file
 * @param: v, a vector of Items.
 * Precondition: v contains the Items to be output to a file.
 * Postcondition: v's values have been written to the file
 *         at the appropriate offsets for this PE.
 * Note: It would be cleaner to pass mpiType as a template parameter
 *         but doing so produces errors (at least for OpenMPI and clang),
 *         so this is a hack-ey workaround.
 *       Could instead pass it as a parameter to the constructor...
 */
template <class ItemType>
void MPIProcessWriter<ItemType>::writeChunk(const std::vector<ItemType> &v)
{
    MPI_File_set_size(OO_IO_Base<ItemType>::getFileHandle(), 0); // truncate

    long chunkSize = v.size();
    OO_IO_Base<ItemType>::setChunkSize(chunkSize);

    long totalItems;
    MPI_Allreduce(&chunkSize, &totalItems, 1,         // find total #
                  MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //  of Items

    OO_IO_Base<ItemType>::setNumItemsInFile(totalItems);
    int itemSize = sizeof(ItemType);
    long totalBytes = totalItems * itemSize;
    OO_IO_Base<ItemType>::setFileSize(totalBytes);
    long start = 0, stop = 0;
    getChunkStartStopValues(OO_IO_Base<ItemType>::getID(),
                            OO_IO_Base<ItemType>::getNumPEs(), totalItems,
                            start, stop);

    OO_IO_Base<ItemType>::setFirstItemOffset(start);
    OO_IO_Base<ItemType>::setFirstByteOffset(start * itemSize);
    MPI_Status status;
    unsigned long itemsWritten = 0;
    unsigned long itemsToWrite = chunkSize;
    int writeResult = 0;
    while (itemsToWrite > INT_MAX)
    {
        writeResult =
            MPI_File_write_at(OO_IO_Base<ItemType>::getFileHandle(),
                              OO_IO_Base<ItemType>::getFirstByteOffset() +
                                  static_cast<MPI_Offset>(itemsWritten) *
                                      itemSize,
                              v.data() + itemsWritten, INT_MAX,
                              OO_IO_Base<ItemType>::getMPIType(), &status);
        checkResult(writeResult);
        itemsWritten += INT_MAX;
        itemsToWrite -= INT_MAX;
    }

    writeResult =
        MPI_File_write_at(OO_IO_Base<ItemType>::getFileHandle(),
                          OO_IO_Base<ItemType>::getFirstByteOffset() +
                              static_cast<MPI_Offset>(itemsWritten) * itemSize,
                          v.data() + itemsWritten, itemsToWrite,
                          OO_IO_Base<ItemType>::getMPIType(), &status);
    checkResult(writeResult);
}

/********************************************************************
 * ThreadsIO opens a file (read-only) for the shared-memory thread
 *  backends and records its size.  It owns the single POSIX file
 *  descriptor that the worker threads read through with pread().
 *
 * It does NOT create threads, synchronize them, or schedule chunks;
 *  those are the responsibility of the user.
 *
 ********************************************************************/

template <class ItemType>
class ThreadsIO : public OO_IO_Base<ItemType>
{
public:
    ThreadsIO(const std::string &fileName, int id, int num_threads,
              int openMode);

    virtual ~ThreadsIO();

  protected:
    int fileDescriptor = -1; 
};

/* ThreadsIO constructor
 * @param: fileName, a string
 * @param: id, an int
 * @param: numThreads, an int
 * @param: openMode, an int
 *          (e.g. O_RDONLY for readers,
 *           O_WRONLY | O_CREAT | O_TRUNC for writers).
 * Precondition : fileName is the name of the file containing binary-format
 * values of type ItemType.
 *          &&  id is the thread id
 *          &&  numThreads is the number of threads
 *          &&  openMode holds valid POSIX open mode flags. (e.g. O_RDONLY for
 * readers) Postcondition: the file is open read-only, its size and item count
 * have been recorded, and fileDescriptor is valid.
 */
template <class ItemType>
ThreadsIO<ItemType>::ThreadsIO(const std::string &fileName, int id,
                               int num_threads, int openMode)
    : OO_IO_Base<ItemType>(fileName, id, num_threads) {
    //  Open the file once, on the constructing thread, with the caller-
    //  supplied mode.  The returned descriptor will be shared by every
    //  worker;
    //  0644 parameter are the file permissions used only when O_CREAT is in
    //  openMode
    fileDescriptor = open(fileName.c_str(), openMode, 0644);

    if (fileDescriptor == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    // Check the open descriptor's metadata for the file's size in bytes.
    struct stat fileInfo;
    // Without the file size we cannot partition the file across the worker
    //  threads; on failure, report the OS error and abort.
    if (fstat(fileDescriptor, &fileInfo) == -1) {
        perror("fstat");
        exit(EXIT_FAILURE);
    }

        // Get file size in bytes for ThreadReader
        long fileSize = fileInfo.st_size;
        OO_IO_Base<ItemType>::setFileSize(fileSize);

        // Compute number of items in file
        OO_IO_Base<ItemType>::setNumItemsInFile(
            fileSize / OO_IO_Base<ItemType>::getItemSize()
        );
    

    // All threads set this to true once the barrier above is cleared
    OO_IO_Base<ItemType>::setFileOpened(true);
}

/* ThreadsIO destructor: closes the shared descriptor if it is open. */
template <class ItemType>
ThreadsIO<ItemType>::~ThreadsIO()
{
    if (fileDescriptor != -1)
    {
        close(fileDescriptor);
    }
}

/********************************************************************
 * The ThreadReader template provides an abstraction to hide the details of
 * threads parallel input.
 *
 * readChunk() works entirely in
 *  locals and reads through pread(), which takes an explicit offset and
 *  so is safe to call concurrently on one shared descriptor.  This is what
 *  makes the thread backends data-race-free.
 ********************************************************************/
template <class ItemType>
class ThreadReader : public ThreadsIO<ItemType>
{
public:
    ThreadReader(const std::string &fileName, int id, int num_threads);
    std::vector<ItemType> readChunk();
    std::vector<ItemType> readChunkPlus(unsigned numExtras);
};

/* ThreadReader constructor: see ThreadsIO Constructor. */
template <class ItemType>
ThreadReader<ItemType>::ThreadReader(const std::string &fileName, int id,
                                     int num_threads)
    : ThreadsIO<ItemType>(fileName, id, num_threads, O_RDONLY) {}

/* Reads the chunk belonging to one worker thread.
 * Return: a vector containing the values of this PE's chunk.
 */
template <class ItemType>
std::vector<ItemType> ThreadReader<ItemType>::readChunk()
{
    long start = 0, stop = 0;

    getChunkStartStopValues(OO_IO_Base<ItemType>::getID(),
                            OO_IO_Base<ItemType>::getNumPEs(),
                            static_cast<unsigned>(OO_IO_Base<ItemType>::
                                                      getNumItemsInFile()),
                            start, stop);

    OO_IO_Base<ItemType>::setChunkSize(stop - start);
    OO_IO_Base<ItemType>::setFirstItemOffset(start);
    OO_IO_Base<
        ItemType>::setFirstByteOffset(start *
                                      OO_IO_Base<ItemType>::getItemSize());

    unsigned long itemsToRead =
        static_cast<unsigned long>(OO_IO_Base<ItemType>::getChunkSize());

    std::vector<ItemType> chunk(itemsToRead);

    off_t byteOffset =
        static_cast<off_t>(OO_IO_Base<ItemType>::getFirstByteOffset());

    size_t expectedBytes = itemsToRead * sizeof(ItemType);

    ssize_t bytesRead = pread(ThreadsIO<ItemType>::fileDescriptor, chunk.data(),
                              expectedBytes, byteOffset);

    if (bytesRead == -1)
    {
        perror("pread");
        exit(EXIT_FAILURE);
    }

    //  A short or failed read means the chunk is incomplete; report the size
    //  mismatch and abort.
    if (bytesRead != static_cast<ssize_t>(expectedBytes))
    {
        fprintf(stderr, "Thread %d: expected %zu bytes, got %zd bytes\n",
                OO_IO_Base<ItemType>::getID(), expectedBytes, bytesRead);
        exit(EXIT_FAILURE);
    }

    return chunk;
}

/* method to read a chunk from the file (in its entirety)
 *  plus a few items from the next thread's chunk.
 *  This version is useful for search problems where the sought-after
 *  value straddles the boundary between two chunks.
 * @param: numExtras, an unsigned.
 * Precondition: numExtras is the number of additional Items to be read
 *                beyond the end of this thread's chunk.
 * Return: a vector containing the values of this thread's chunk
 *          plus numExtras values of the next thread's chunk
 *          for all threads except the last one.
 *
 * Note: Giving numExtras a default parameter value would let us eliminate
 *        the other readChunk() method and adhere to the DRY principle,
 *        but defining each separately lets us more clearly explain
 *        the difference between the two versions.
 */
template <class ItemType>
std::vector<ItemType>
ThreadReader<ItemType>::readChunkPlus(unsigned numExtras)
{
    // Note: the file's size and item count were already recorded by the
    //  ThreadsIO constructor (via fstat), so unlike the MPI version we
    //  do not re-stat the file here.
    long numItemsInFile = OO_IO_Base<ItemType>::getNumItemsInFile();
    long start = 0, stop = 0;
    int id = OO_IO_Base<ItemType>::getID();
    int numPEs = OO_IO_Base<ItemType>::getNumPEs();

    getChunkStartStopValues(id, numPEs, numItemsInFile, start, stop);

    // Extend this thread's chunk by numExtras Items, except for the last
    //  thread, then clamp so we never read past the end of the file.
    if (id < numPEs - 1)
    {
        stop += numExtras;
    }
    if (stop > numItemsInFile)
    {
        stop = numItemsInFile;
    }

    OO_IO_Base<ItemType>::setChunkSize(stop - start);
    OO_IO_Base<ItemType>::setFirstItemOffset(start);
    OO_IO_Base<
        ItemType>::setFirstByteOffset(start *
                                      OO_IO_Base<ItemType>::getItemSize());

    unsigned long itemsToRead =
        static_cast<unsigned long>(OO_IO_Base<ItemType>::getChunkSize());

    std::vector<ItemType> chunk(itemsToRead);

    off_t byteOffset =
        static_cast<off_t>(OO_IO_Base<ItemType>::getFirstByteOffset());

    size_t expectedBytes = itemsToRead * sizeof(ItemType);

    ssize_t bytesRead = pread(ThreadsIO<ItemType>::fileDescriptor, chunk.data(),
                              expectedBytes, byteOffset);

    if (bytesRead == -1)
    {
        perror("pread");
        exit(EXIT_FAILURE);
    }

    // A short or failed read means the chunk is incomplete; report the
    //  size mismatch and abort.
    if (bytesRead != static_cast<ssize_t>(expectedBytes))
    {
        fprintf(stderr, "Thread %d: expected %zu bytes, got %zd bytes\n",
                OO_IO_Base<ItemType>::getID(), expectedBytes, bytesRead);
        exit(EXIT_FAILURE);
    }

    return chunk;
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
class ThreadWriter : public ThreadsIO<ItemType>
{
public:
    ThreadWriter(const std::string &fileName, int id, int num_threads, long fileSize);
    void writeChunk(const std::vector<ItemType> &v);
};

/* ThreadWriter constructor
 * @param: fileName, a string
 * @param: id, an int
 * @param: numThreads, an int
 * @param: fileSize, an int
 * Precondition : fileName is the name of the file containing binary-format
 * values of type ItemType.
 *          &&  id is the thread id
 *          &&  numThreads is the number of threads
 *          &&  fileSize is the size of the file read in bytes
 */
template <class ItemType>
ThreadWriter<ItemType>::ThreadWriter(const std::string &fileName, int id,
                                     int num_threads, long fileSize)
    : ThreadsIO<ItemType>(fileName, id, num_threads, O_WRONLY | O_CREAT | O_TRUNC)
{
    OO_IO_Base<ItemType>::setFileSize(fileSize);
}

/* method to write this thread's chunk to the file
 * @param: v, a vector of Items
 * Precondition: v contains the Items assigned to this thread
 *            && all threads collectively hold the full dataset
 *            && threads cooperate so their chunks do not overlap
 * Postcondition: v's values have been written to the file
 *                at the correct offset for this thread
 *             && the full dataset has been written without gaps or overlap
 *
 *       pwrite() is used so each thread writes directly to its
 *       assigned file position without interfering with others.
 */

template <class ItemType>
void ThreadWriter<ItemType>::writeChunk(const std::vector<ItemType> &v)
{
    // Total items in file (given by user via fileSize)
    long totalItems =
        OO_IO_Base<ItemType>::getFileSize() / sizeof(ItemType);

    OO_IO_Base<ItemType>::setNumItemsInFile(totalItems);

    // Compute chunk boundaries
    long start = 0, stop = 0;
    getChunkStartStopValues(
        OO_IO_Base<ItemType>::getID(),
        OO_IO_Base<ItemType>::getNumPEs(),
        totalItems,
        start, stop);

    long expectedChunkSize = stop - start;
    long actualChunkSize = static_cast<long>(v.size());

    // Safety check: ensure correct partitioning
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

    off_t byteOffset =
        static_cast<off_t>(OO_IO_Base<ItemType>::getFirstByteOffset());

    size_t bytesToWrite = actualChunkSize * sizeof(ItemType);

    // Robust write loop (handles partial writes)
    size_t totalWritten = 0;
    const char* data = reinterpret_cast<const char*>(v.data());  // view data as raw bytes for pwrite.

    while (totalWritten < bytesToWrite)
    {
        ssize_t written = pwrite(
            ThreadsIO<ItemType>::fileDescriptor,
            data + totalWritten,
            bytesToWrite - totalWritten,
            byteOffset + totalWritten);

        if (written <= 0)
        {
            perror("pwrite");
            exit(EXIT_FAILURE);
        }

        totalWritten += written;
    }
}

    /**                          HELPER UTILITIES
     * -------------------------------------------------------------------------
     * Place any small, header-safe helper functions here.
     * All functions MUST be declared inline to avoid multiple-definition errors
     * when this header is included in multiple translation units.
     * ------------------------------------------------------------------------- */

    /* ----------------------------------------------------------------------
     * mpiType<T>() returns the MPI_Datatype corresponding to a C++ type T.
     * Specializations are defined here, before any template that uses them.
     * Add a new specialization to support an additional ItemType.
     * -------------------------------------------------------------------- */
    template <>
    inline MPI_Datatype mpiType<int>()
{
    return MPI_INT;
}
template <>
inline MPI_Datatype mpiType<long>() { return MPI_LONG; }
template <>
inline MPI_Datatype mpiType<float>() { return MPI_FLOAT; }
template <>
inline MPI_Datatype mpiType<double>() { return MPI_DOUBLE; }
template <>
inline MPI_Datatype mpiType<char>() { return MPI_CHAR; }

/* Utility to check the return-values of MPI-IO function calls
 * @param: result, an int
 * Precondition:  result is the return-value from the last MPI-IO call.
 * Postcondition: If result is anything other than MPI_SUCCESS::
 *                 the string associated with result has been printed to stderr
 *                 && the program has been terminated abnormally.
 */
inline void checkResult(int result)
{
    if (result != MPI_SUCCESS)
    {
        char errorString[1024] = {'\0'};
        int errorStringLength = -1;
        int errorClass = -1;

        MPI_Error_class(result, &errorClass);
        MPI_Error_string(errorClass, errorString, &errorStringLength);
        fprintf(stderr, "\nMPI Error: '%s'\n\n", errorString);

        MPI_Abort(MPI_COMM_WORLD, result);
    }
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
inline void getChunkStartStopValues(int id, int numPEs, const unsigned REPS,
                                    long &start, long &stop)
{
    // check precondition before proceeding
    if ((unsigned)numPEs > REPS)
    {
        if (id == 0)
        {
            printf("\n*** Number of PEs (%u) exceeds REPS (%u)\n", numPEs,
                   REPS);
            printf("*** Please run using PEs less than or equal to %u\n\n",
                   REPS);
        }
        exit(EXIT_FAILURE);
        return;
    }

    // compute the chunk size that works in many cases
    unsigned chunkSize1 = (REPS + numPEs - 1) / numPEs; // integer ceiling
    unsigned begin = id * chunkSize1;
    unsigned end = begin + chunkSize1;
    // see if there are any leftover iterations
    unsigned remainder = REPS % numPEs;
    // If remainder == 0, chunkSize1 = chunk-size for all PEs;
    // If remainder != 0, chunkSize1 = chunk-size for p_0..p_remainder-1
    //   but for PEs p_remainder..p_numPEs-1
    //   recompute begin and end using a smaller-by-1 chunk size, chunkSize2.
    if (remainder > 0 && (unsigned)id >= remainder)
    {
        unsigned chunkSize2 = chunkSize1 - 1;
        unsigned remainderBase = remainder * chunkSize1;
        unsigned peOffset = (id - remainder) * chunkSize2;
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