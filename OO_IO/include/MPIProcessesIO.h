/* MPIProcessesIO.h conains the MPIProcessesIO class that supports
 * parallel binary I/O using MPI-IO (processes).
 *
 * @author: Joshua Eilu for Professor Joel Adams at Calvin University.
 * @date:   Summer 2026
 */

#ifndef OO_IO_MPI_PROCESS_IO_H
#define OO_IO_MPI_PROCESS_IO_H

#include "IO_Base.h"
#include <mpi.h>

/********************************************************************
 * Helper function declarations.
 ********************************************************************/
template <typename T>
inline MPI_Datatype mpiType();
inline void checkResult(int result);

/* ----------------------------------------------------------------------
 * MPI_Runtime: a single, process-wide MPI initializer/finalizer.
 *
 * MPI_Init() and MPI_Finalize() must each be called exactly once per
 *  process.  A function-local static object (see mpiRuntime() below)
 * is guaranteed by C++ to be constructed exactly once, which lets
 * this library manage MPI's lifetime without the user calling
 * MPI_Init() or MPI_Finalize() explicitly.
 * --------------------------------------------------------------------
 */
class MPI_Runtime {
  public:
    MPI_Runtime() {
        int initialized = 0;
        MPI_Initialized(&initialized);

        if (!initialized) {
            // MPI Initialization
            MPI_Init(nullptr, nullptr);
            initializedMPI = true;
        }

        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        MPI_Comm_size(MPI_COMM_WORLD, &numPEs);
    }

    ~MPI_Runtime() {
        int finalized = 0;
        MPI_Finalized(&finalized);

        // Only finalize MPI if:
        // 1. We initialized it, AND
        // 2. It has not already been finalized.
        if (initializedMPI && !finalized) {
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

// Ensures a single process-wide MPI runtime (init/finalize).
inline MPI_Runtime &mpiRuntime() {
    static MPI_Runtime instance;
    return instance;
}

/* MPIProcessesIO is a templated base class that supports parallel
 * binary I/O using MPI-IO (processes). It stores information about
 * the file and each PE’s chunk, and provides helper functions for
 * derived classes.
 *
 * It's subclasses are MPIProcessReader and MPIProcessWriter.
 */
template <class ItemType>
class MPIProcessesIO : public IO_Base<ItemType> {
  public:
    MPIProcessesIO(int mpiMode);
    virtual ~MPIProcessesIO() = default;

    void open(const std::string &fileName) override; // open and close the corresponding
    void close() override; // file for reading or writing using MPI-IO.

    MPI_File &getFileHandle() { return myFileHandle; }
    MPI_Datatype getMPIType() const { return myMPIType; }

  private:
    MPI_Datatype myMPIType; // the MPI equivalent of ItemType
    MPI_File myFileHandle;  // MPI handle (MPI backend only)
    int mpiMode;            // MPI file open mode (MPI backend only)
};

/* MPIProcessesIO constructor
 * @param: mpiMode, an int
 * @param: mpiType, an MPI_Datatype
 * Precondition: fileName is the name of a file containing
 * binary-format values of type ItemType. Postcondition: the file has
 * been opened for parallel input or output
 *             &&  each instance variable have been initialized
 *                  as appropriate for this PE using the file's info.
 */
template <class ItemType>
MPIProcessesIO<ItemType>::MPIProcessesIO(int mpiMode)
    : IO_Base<ItemType>(mpiRuntime().getRank(), mpiRuntime().getNumPEs()),
      myMPIType(mpiType<ItemType>()), myFileHandle(MPI_FILE_NULL), mpiMode(mpiMode) {}

/* MPIProcessesIO::open() opens the file for parallel input or output
 * using MPI-IO.
 * @param: fileName, a string
 * Precondition: fileName is the name of a file containing
 * binary-format values of type ItemType. Postcondition: the file has
 * been opened for parallel input or output
 *             &&  each instance variable have been initialized
 *                  as appropriate for this PE using the file's info.
 */
template <class ItemType>
void MPIProcessesIO<ItemType>::open(const std::string &fileName) {
    // Open the file for parallel input or output using MPI-IO
    int result = MPI_File_open(MPI_COMM_WORLD, fileName.c_str(), mpiMode, MPI_INFO_NULL,
                               &myFileHandle);
    checkResult(result);
    IO_Base<ItemType>::setFileName(fileName);
    IO_Base<ItemType>::setFileOpened(true);
}

/* MPIProcessesIO::close() closes the file for parallel input or
 * output using MPI-IO. Precondition: the file has been opened for
 * parallel input or output Postcondition: the file has been closed
 * for parallel input or output
 */
template <class ItemType>
void MPIProcessesIO<ItemType>::close() {

    if (!IO_Base<ItemType>::getFileOpened()) {
        fprintf(stderr, "\nMPIProcessesIO::close(): file was not opened\n\n");
        exit(EXIT_FAILURE);
    }
    // Close the file for parallel input or output using MPI-IO
    int result = MPI_File_close(&myFileHandle);
    checkResult(result);
    IO_Base<ItemType>::setFileOpened(false);
}

/* MPIProcessReader template provides an abstraction to hide the
 * details of MPI-IO parallel input.
 */
template <class ItemType>
class MPIProcessReader : public MPIProcessesIO<ItemType> {
  public:
    MPIProcessReader();
    std::vector<ItemType> readChunk();
    std::vector<ItemType> readChunkPlus(unsigned numExtras);
    virtual ~MPIProcessReader() = default;
};

/* MPIProcessReader Constructor */
template <class ItemType>
MPIProcessReader<ItemType>::MPIProcessReader()
    : MPIProcessesIO<ItemType>(MPI_MODE_RDONLY) {}

/* method to read a chunk from the file (in its entirety).
 * Return: a vector containing the values of this PE's chunk.
 * Note: vector was chosen as the return-type because it
 *        uses contiguous memory and provides a move-constructor.
 */
template <class ItemType>
std::vector<ItemType> MPIProcessReader<ItemType>::readChunk() {
    // Note: We could compute the following attributes in the
    // constructor,
    //  but do them here for symmetry with MPIProcessWriter
    MPI_Offset fileSize;
    MPI_File_get_size(this->getFileHandle(), &fileSize);
    IO_Base<ItemType>::setFileSize(fileSize);
    // Note: EOF char seems inconsistent on different platforms;
    //  if char tests fail and off-by-one, uncomment the next 3 lines
    //   if (std::is_same<ItemType, char>::value) {         // if
    //   ItemType is char
    //      --fileSize;                                     // ignore
    //      EOF char
    //   }
    IO_Base<ItemType>::setNumItemsInFile(fileSize / IO_Base<ItemType>::getItemSize());

    long start = 0, stop = 0;
    getChunkStartStopValues(IO_Base<ItemType>::getID(), IO_Base<ItemType>::getNumPEs(),
                            IO_Base<ItemType>::getNumItemsInFile(), start, stop);
    IO_Base<ItemType>::setChunkSize(stop - start);
    IO_Base<ItemType>::setFirstItemOffset(start);
    IO_Base<ItemType>::setFirstByteOffset(start * IO_Base<ItemType>::getItemSize());

    MPI_Status status;
    unsigned long itemsRead = 0;
    unsigned long itemsToRead = IO_Base<ItemType>::getChunkSize();
    std::vector<ItemType> v(itemsToRead);
    int readResult = 0;
    // handle very large files where chunkSize > INT_MAX
    while (itemsToRead > INT_MAX) {
        readResult = MPI_File_read_at(
            this->getFileHandle(), IO_Base<ItemType>::getFirstByteOffset() + itemsRead,
            v.data() + itemsRead, INT_MAX, this->getMPIType(), &status);
        checkResult(readResult);
        itemsRead += INT_MAX;
        itemsToRead -= INT_MAX;
    }
    // read in remaining Items (or if itemsToRead <= INT_MAX
    // initially)
    readResult = MPI_File_read_at(
        this->getFileHandle(), IO_Base<ItemType>::getFirstByteOffset() + itemsRead,
        v.data() + itemsRead, itemsToRead, this->getMPIType(), &status);
    checkResult(readResult);

    return v;
}

/* method to read a chunk from the file (in its entirety)
 *  plus a few items from the next PE's chunk.
 *  This version of the method is useful for search problems where
 *  the sought-after value straddles the boundary between two chunks.
 * @param: numExtras, an unsigned.
 * Precondition: numExtras is the number of additional Items to be
 * read beyond the end of this PE's chunk. Return: a vector containing
 * the values of this PE's chunk plus numExtras values of the next
 * PE's chunk for all PEs except the last one.
 *
 * Note: Giving numExtras a default parameter value would let us
 * eliminate the other readChunk() method and adhere to the DRY
 * principle, but defining each separately lets us more clearly
 * explain the difference between the two versions.
 */
template <class ItemType>
std::vector<ItemType> MPIProcessReader<ItemType>::readChunkPlus(unsigned numExtras) {
    // Note: We could compute the following attributes in the
    // constructor,
    //  but do them here for symmetry with MPIProcessWriter
    MPI_Offset fileSize;
    MPI_File_get_size(this->getFileHandle(), &fileSize);
    IO_Base<ItemType>::setFileSize(fileSize);
    // Note: EOF char seems inconsistent on different platforms;
    //  if char tests fail and off-by-one, uncomment the next 3 lines
    //   if (std::is_same<ItemType, char>::value) {         // if
    //   ItemType is char
    //      --fileSize;                                     // ignore
    //      EOF char
    //   }
    IO_Base<ItemType>::setNumItemsInFile(fileSize / IO_Base<ItemType>::getItemSize());

    long numItemsInFile = IO_Base<ItemType>::getNumItemsInFile();
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
    IO_Base<ItemType>::setChunkSize(stop - start);
    IO_Base<ItemType>::setFirstItemOffset(start);
    IO_Base<ItemType>::setFirstByteOffset(start * IO_Base<ItemType>::getItemSize());

    MPI_Status status;
    unsigned long itemsRead = 0;
    unsigned long itemsToRead = IO_Base<ItemType>::getChunkSize();
    std::vector<ItemType> v(itemsToRead);
    int readResult = 0;
    // handle very large files where chunkSize > INT_MAX
    while (itemsToRead > INT_MAX) {
        readResult = MPI_File_read_at(
            this->getFileHandle(), IO_Base<ItemType>::getFirstByteOffset() + itemsRead,
            v.data() + itemsRead, INT_MAX, this->getMPIType(), &status);
        checkResult(readResult);
        itemsRead += INT_MAX;
        itemsToRead -= INT_MAX;
    }
    // read in remaining Items (or if itemsToRead <= INT_MAX
    // initially)
    readResult = MPI_File_read_at(
        this->getFileHandle(), IO_Base<ItemType>::getFirstByteOffset() + itemsRead,
        v.data() + itemsRead, itemsToRead, this->getMPIType(), &status);
    checkResult(readResult);

    return v;
}

/* MPIProcessWriter template provides an abstraction to hide the
 * details of MPI-IO parallel output.
 */
template <class ItemType>
class MPIProcessWriter : public MPIProcessesIO<ItemType> {
  public:
    MPIProcessWriter();
    void writeChunk(const std::vector<ItemType> &v);
    virtual ~MPIProcessWriter() = default;
};

/* MPIProcessWriter Constructor */
template <class ItemType>
MPIProcessWriter<ItemType>::MPIProcessWriter()
    : MPIProcessesIO<ItemType>(MPI_MODE_WRONLY | MPI_MODE_CREATE) {}

/* method to write this PE's chunk to the file
 * @param: v, a vector of Items.
 * Precondition: v contains the Items to be output to a file.
 * Postcondition: v's values have been written to the file
 *         at the appropriate offsets for this PE.
 */
template <class ItemType>
void MPIProcessWriter<ItemType>::writeChunk(const std::vector<ItemType> &v) {
    MPI_File_set_size(this->getFileHandle(), 0); // truncate

    long chunkSize = v.size();
    IO_Base<ItemType>::setChunkSize(chunkSize);

    long totalItems;
    MPI_Allreduce(&chunkSize, &totalItems, 1,         // find total #
                  MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //  of Items

    IO_Base<ItemType>::setNumItemsInFile(totalItems);
    int itemSize = sizeof(ItemType);
    long totalBytes = totalItems * itemSize;
    IO_Base<ItemType>::setFileSize(totalBytes);
    long start = 0, stop = 0;
    getChunkStartStopValues(IO_Base<ItemType>::getID(), IO_Base<ItemType>::getNumPEs(),
                            totalItems, start, stop);

    IO_Base<ItemType>::setFirstItemOffset(start);
    IO_Base<ItemType>::setFirstByteOffset(start * itemSize);
    MPI_Status status;
    unsigned long itemsWritten = 0;
    unsigned long itemsToWrite = chunkSize;
    int writeResult = 0;
    while (itemsToWrite > INT_MAX) {
        writeResult = MPI_File_write_at(
            this->getFileHandle(),
            IO_Base<ItemType>::getFirstByteOffset() +
                static_cast<MPI_Offset>(itemsWritten) * itemSize,
            v.data() + itemsWritten, INT_MAX, this->getMPIType(), &status);
        checkResult(writeResult);
        itemsWritten += INT_MAX;
        itemsToWrite -= INT_MAX;
    }

    writeResult = MPI_File_write_at(this->getFileHandle(),
                                    IO_Base<ItemType>::getFirstByteOffset() +
                                        static_cast<MPI_Offset>(itemsWritten) * itemSize,
                                    v.data() + itemsWritten, itemsToWrite,
                                    this->getMPIType(), &status);
    checkResult(writeResult);
}

/**                          HELPER UTILITIES
 * -------------------------------------------------------------------------
 * Place any small, header-safe helper functions here.
 * All functions MUST be declared inline to avoid multiple-definition
 * errors when this header is included in multiple translation units.
 * -------------------------------------------------------------------------
 */

/* ----------------------------------------------------------------------
 * mpiType<T>() returns the MPI_Datatype corresponding to a C++ type
 * T. Specializations are defined here, before any template that uses
 * them. Add a new specialization to support an additional ItemType.
 * --------------------------------------------------------------------
 */
template <>
inline MPI_Datatype mpiType<int>() {
    return MPI_INT;
}
template <>
inline MPI_Datatype mpiType<long>() {
    return MPI_LONG;
}
template <>
inline MPI_Datatype mpiType<float>() {
    return MPI_FLOAT;
}
template <>
inline MPI_Datatype mpiType<double>() {
    return MPI_DOUBLE;
}
template <>
inline MPI_Datatype mpiType<char>() {
    return MPI_CHAR;
}

/* Utility to check the return-values of MPI-IO function calls
 * @param: result, an int
 * Precondition:  result is the return-value from the last MPI-IO
 * call. Postcondition: If result is anything other than MPI_SUCCESS::
 *                 the string associated with result has been printed
 * to stderr
 *                 && the program has been terminated abnormally.
 */
inline void checkResult(int result) {
    if (result != MPI_SUCCESS) {
        char errorString[1024] = {'\0'};
        int errorStringLength = -1;
        int errorClass = -1;

        MPI_Error_class(result, &errorClass);
        MPI_Error_string(errorClass, errorString, &errorStringLength);
        fprintf(stderr, "\nMPI Error: '%s'\n\n", errorString);

        MPI_Abort(MPI_COMM_WORLD, result);
    }
}

#endif // OO_IO_MPI_PROCESS_IO_H