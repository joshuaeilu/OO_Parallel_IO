/* threadTests.cpp contains tests for reading and writing files using threads.
 *
 * @author: Joshua Eilu for Joel C. Adams,
 *        Calvin University, Summer 2026
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../OO_IO/include/ThreadsIO.h"

#include <barrier>
#include <memory>
#include <pthread.h>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <omp.h>


/* ThreadTestInstance
 * Stores information about a thread participating in a test.
 *
 * Provides thread identification, the total number of participating
 * threads, and a framework-independent synchronization operation.
 */
struct ThreadTestInstance {
    int threadID = 0;
    int numThreads = 1;
    std::function<void()> synchronizeFunction;

    /**
     * Synchronize the current thread with other participating threads.
     */
    void synchronize() const {
        synchronizeFunction();
    }
};

/*
 * Sets up a ThreadTestInstance for OpenMP threads.
 */
ThreadTestInstance setupOpenMPThreadTestInstance(std::barrier<> &barrier) {
    ThreadTestInstance instance;
    instance.threadID = omp_get_thread_num();
    instance.numThreads = omp_get_num_threads();
    instance.synchronizeFunction = [&barrier]() {
        barrier.arrive_and_wait();
    };
    return instance;
}

/*
 * Sets up a ThreadTestInstance for standard threads.
 */
ThreadTestInstance setupStdThreadTestInstance(
    int threadID,
    int numThreads,
    std::barrier<> &barrier) {

    ThreadTestInstance instance;

    instance.threadID = threadID;
    instance.numThreads = numThreads;
    instance.synchronizeFunction = [&barrier]() {
        barrier.arrive_and_wait();
    };

    return instance;
}

/**
 * @brief Runs a test function using standard C++ threads.
 * @param numThreads The number of threads to create.
 * @param testFunction The testing function executed by each thread.
 */
void runWithStdThreads(
    int numThreads,
    const std::function<void(const ThreadTestInstance &)> &testFunction) {

    std::barrier barrier(numThreads);
    std::vector<std::thread> threads;

    for (int threadID = 0; threadID < numThreads; ++threadID) {
        threads.emplace_back([&, threadID]() {
            ThreadTestInstance instance =
                setupStdThreadTestInstance(threadID, numThreads, barrier);

            testFunction(instance);
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }
}

/**
 * @brief Stores the arguments required by a POSIX thread.
 */
struct PthreadTestContext {
    int threadID;
    int numThreads;
    pthread_barrier_t *barrier;

    const std::function<void(const ThreadTestInstance &)> *testFunction;
};

/**
 * @brief Creates a ThreadTestInstance for a POSIX thread.
 * @param context The context containing thread information and its barrier.
 * @return The initialized thread test instance.
 */
ThreadTestInstance setupPthreadTestInstance(const PthreadTestContext &context) {
    ThreadTestInstance instance;

    instance.threadID = context.threadID;
    instance.numThreads = context.numThreads;

    instance.synchronizeFunction = [barrier = context.barrier]() {
        const int result = pthread_barrier_wait(barrier);

        if (result != 0 && result != PTHREAD_BARRIER_SERIAL_THREAD) {
            throw std::runtime_error("pthread_barrier_wait failed");
        }
    };

    return instance;
}

/**
 * @brief Entry point for a POSIX worker thread.
 * @param argument Pointer to the thread's test context.
 * @return nullptr after the testing function completes.
 */
void *pthreadTestWorker(void *argument) {
    auto *context = static_cast<PthreadTestContext *>(argument);

    ThreadTestInstance instance = setupPthreadTestInstance(*context);

    (*context->testFunction)(instance);

    return nullptr;
}

/**
 * @brief Runs a test function using POSIX threads.
 * @param numThreads The number of threads to create.
 * @param testFunction The testing function executed by each thread.
 */
void runWithPthreads(
    int numThreads,
    const std::function<void(const ThreadTestInstance &)> &testFunction) {

    pthread_barrier_t barrier;

    if (pthread_barrier_init(&barrier, nullptr, numThreads) != 0) {
        throw std::runtime_error("pthread_barrier_init failed");
    }

    std::vector<pthread_t> threads(numThreads);
    std::vector<PthreadTestContext> contexts(numThreads);

    int createdThreads = 0;

    for (int i = 0; i < numThreads; ++i) {
        contexts[i] = {
            i,
            numThreads,
            &barrier,
            &testFunction
        };

        if (pthread_create(
                &threads[i],
                nullptr,
                pthreadTestWorker,
                &contexts[i]) != 0) {
            break;
        }

        ++createdThreads;
    }

    if (createdThreads != numThreads) {
        // Do not leave successfully created workers waiting at a barrier
        // that expects threads that were never created.
        std::terminate();
    }

    for (auto &thread : threads) {
        pthread_join(thread, nullptr);
    }

    pthread_barrier_destroy(&barrier);
}




// Checks if two double values are approximately equal.
bool approximatelyEqual(double v1, double v2) {
    const double THRESHOLD = 0.0000000000001;

    return std::abs(v2 - v1) < THRESHOLD;
}

/*
 * Converts each C++ datatype into the word used in the corresponding test filename.
 * This is used to generate the appropriate filename for each data type.
 * @return: The corresponding filename word for the given C++ datatype.
 */
template <typename Type>
constexpr const char *getTypeFileName() {
    if constexpr (std::is_same_v<Type, double>) {
        return "doubles";
    } else if constexpr (std::is_same_v<Type, int>) {
        return "ints";
    } else if constexpr (std::is_same_v<Type, char>) {
        return "chars";
    }
}

const int threadCounts[] = {1, 2, 4, 8};  // Array of thread and
const int itemCounts[] = {1, 3, 4, 8, 9}; // item counts to test with

/*
 * The four test getter functions below test the getter methods of ThreadReader and ThreadWriter
 * classes for both minimal and file-based constructors before any file operations are performed.
 * e.g. a file is being read or written. This ensures that the instance variables of the reader and
 * writer objects are correctly initialized before any file operations take place.
 */
void testGetterFunctionsForReadersMinimalConstructor(const ThreadTestInstance &instance) {
    ThreadReader<double> reader(instance.threadID, instance.numThreads);

    CHECK(reader.getID() == instance.threadID);

    CHECK(reader.getNumPEs() == instance.numThreads);

    CHECK(reader.getFileName() == "");

    CHECK(reader.getItemSize() == sizeof(double));

    CHECK_FALSE(reader.getFileOpened());
}
void testGetterFunctionsForReadersFileBasedConstructor(const ThreadTestInstance &instance) {
    const std::string fileName = "./files/5doubles.bin";

    ThreadReader<double> reader(instance.threadID, instance.numThreads, fileName);

    CHECK(reader.getID() == instance.threadID);
    CHECK(reader.getNumPEs() == instance.numThreads);
    CHECK(reader.getFileName() == fileName);
    CHECK(reader.getItemSize() == sizeof(double));
    CHECK(reader.getFileOpened());

    reader.close();

    CHECK_FALSE(reader.getFileOpened());
}
void testGetterFunctionsForWritersMinimalConstructor(const ThreadTestInstance &instance) {
    long fileSize = 5L * static_cast<long>(sizeof(double));
    ThreadWriter<double> writer(instance.threadID, instance.numThreads, fileSize);

    CHECK(writer.getID() == instance.threadID);
    CHECK(writer.getNumPEs() == instance.numThreads);
    CHECK(writer.getFileName() == "");
    CHECK(writer.getItemSize() == sizeof(double));
    CHECK_FALSE(writer.getFileOpened());
}
void testGetterFunctionsForWritersFileBasedConstructor(const ThreadTestInstance &instance) {
    const std::string fileName = "./files/writefiletest.bin";
    long fileSize = 5L * static_cast<long>(sizeof(double));
    ThreadWriter<double> writer(instance.threadID, instance.numThreads, fileSize, fileName);

    CHECK(writer.getID() == instance.threadID);
    CHECK(writer.getNumPEs() == instance.numThreads);
    CHECK(writer.getFileName() == fileName);
    CHECK(writer.getItemSize() == sizeof(double));
    CHECK(writer.getFileOpened());

    writer.close();

    CHECK_FALSE(writer.getFileOpened());
}

/*
 * The four test functions below verify the open and close operations for ThreadReader and
 * ThreadWriter classes.
 */
void testReaderOpenCloseMinimalConstructor(const ThreadTestInstance &instance) {
    const std::string fileName = "./files/5doubles.bin";

    ThreadReader<double> reader(instance.threadID, instance.numThreads);

    // Verify initial state.
    CHECK_FALSE(reader.getFileOpened());
    CHECK(reader.getFileName() == "");

    reader.open(fileName, O_RDONLY);

    // Verify state after opening.
    CHECK(reader.getFileName() == fileName);
    CHECK(reader.getItemSize() == sizeof(double));
    CHECK(reader.getFileSize() == 5L * static_cast<long>(sizeof(double)));
    CHECK(reader.getNumItemsInFile() == 5);
    CHECK(reader.getFileOpened());

    // All threads finish checking before closing.
    instance.synchronize();

    reader.close();

    CHECK_FALSE(reader.getFileOpened());
}
void testReaderOpenCloseFileBasedConstructor(const ThreadTestInstance &instance) {
    const std::string fileName = "./files/5doubles.bin";

    ThreadReader<double> reader(instance.threadID, instance.numThreads, fileName);

    CHECK(reader.getFileName() == fileName);
    CHECK(reader.getItemSize() == sizeof(double));
    CHECK(reader.getFileSize() == 5L * static_cast<long>(sizeof(double)));
    CHECK(reader.getNumItemsInFile() == 5);
    CHECK(reader.getFileOpened());

    instance.synchronize();

    reader.close();

    CHECK_FALSE(reader.getFileOpened());
}
// Prepares the output file for writing by removing it if it exists and synchronizing all threads.
void prepareOutputFile(const ThreadTestInstance &instance, const std::string &fileName) {

    if (instance.threadID == 0) {
        std::remove(fileName.c_str());
    }

    instance.synchronize();
}
void testWriterOpenCloseUsingMinimalConstructor(const ThreadTestInstance &instance) {
    const std::string fileName = "./files/test-writer-output.bin";

    const long fileSize = 5L * static_cast<long>(sizeof(double));

    prepareOutputFile(instance, fileName);
    ThreadWriter<double> writer(instance.threadID, instance.numThreads, fileSize);

    // Verify the writer begins unopened.
    CHECK_FALSE(writer.getFileOpened());
    CHECK(writer.getFileName() == "");

    /*
     * Thread 0 creates, resizes, and maps the output file.
     * The other threads wait until that work is complete.
     */
    writer.open(fileName, O_RDWR);

    // Verify that all threads were released from open().
    CHECK(writer.getFileOpened());

    // Verify the output filename was recorded.
    CHECK(writer.getFileName() == fileName);

    // Verify the requested file size was preserved.
    CHECK(writer.getFileSize() == fileSize);

    // Verify the output contains space for five doubles.
    CHECK(writer.getNumItemsInFile() == 5);

    CHECK(writer.getItemSize() == sizeof(double));

    // Nobody closes the shared mapping while another
    // thread is still examining the opened writer.
    instance.synchronize();

    writer.close();

    CHECK_FALSE(writer.getFileOpened());
}
void testWriterOpenCloseUsingFileBasedConstructor(const ThreadTestInstance &instance) {
    const std::string fileName = "./files/test-writer-output.bin";

    const long fileSize = 5L * static_cast<long>(sizeof(double));

    prepareOutputFile(instance, fileName);
    ThreadWriter<double> writer(instance.threadID, instance.numThreads, fileSize, fileName);

    CHECK(writer.getFileOpened());
    CHECK(writer.getFileName() == fileName);
    CHECK(writer.getItemSize() == sizeof(double));
    CHECK(writer.getFileSize() == fileSize);
    CHECK(writer.getNumItemsInFile() == 5);

    instance.synchronize();

    writer.close();

    CHECK_FALSE(writer.getFileOpened());
}

/*
 * The two test functions below verify the getters for variables known after reading a file.
 * This ensures that the getters return correct instance variables for the file's properties after it has been opened
 * and read.
 */
void testGettersForVariablesKnownAfterReadingUsingMinimalConstructor(const ThreadTestInstance &instance) {
    ThreadReader<double> reader(instance.threadID, instance.numThreads);
    reader.open("./files/5doubles.bin", O_RDONLY);
    std::span<const double> chunk = reader.readChunk();

    CHECK(reader.getFileOpened() == true);
    CHECK(reader.getNumItemsInFile() == 5);
    CHECK(reader.getFileSize() == 5 * sizeof(double));
    CHECK(reader.getChunkSize() == 5 / instance.numThreads + (instance.threadID < 5 % instance.numThreads));
    CHECK(reader.getFirstItemOffset() ==
          instance.threadID * (5 / instance.numThreads) + std::min(instance.threadID, 5 % instance.numThreads));
    CHECK(reader.getFirstByteOffset() ==
          (instance.threadID * (5 / instance.numThreads) + std::min(instance.threadID, 5 % instance.numThreads)) *
              sizeof(double));

    instance.synchronize();
    reader.close();
}
void testGettersForVariablesKnownAfterReadingUsingFileBasedConstructor(const ThreadTestInstance &instance) {
    ThreadReader<double> reader(instance.threadID, instance.numThreads, "./files/5doubles.bin");
    std::span<const double> chunk = reader.readChunk();

    CHECK(reader.getFileOpened() == true);
    CHECK(reader.getNumItemsInFile() == 5);
    CHECK(reader.getFileSize() == 5 * sizeof(double));
    CHECK(reader.getChunkSize() == 5 / instance.numThreads + (instance.threadID < 5 % instance.numThreads));
    CHECK(reader.getFirstItemOffset() ==
          instance.threadID * (5 / instance.numThreads) + std::min(instance.threadID, 5 % instance.numThreads));
    CHECK(reader.getFirstByteOffset() ==
          (instance.threadID * (5 / instance.numThreads) + std::min(instance.threadID, 5 % instance.numThreads)) *
              sizeof(double));

    instance.synchronize();
    reader.close();
}

/**
 * The two test functions below verify the behavior of reading an empty file.
 * They ensure that the reader correctly identifies the file as empty and that all related getters return appropriate
 * values.
 */
void testReadingAnEmptyFileUsingMinimalConstructor(const ThreadTestInstance &instance) {
    ThreadReader<double> reader(instance.threadID, instance.numThreads);

    reader.open("./files/empty.bin", O_RDONLY);
    std::span<const double> chunk = reader.readChunk();

    CHECK(reader.getFileOpened());
    CHECK(reader.getNumItemsInFile() == 0);
    CHECK(reader.getFileSize() == 0);
    CHECK(reader.getChunkSize() == 0);
    CHECK(reader.getFirstItemOffset() == 0);
    CHECK(reader.getFirstByteOffset() == 0);

    instance.synchronize();
    reader.close();
}
void testReadingAnEmptyFileUsingFileBasedConstructor(const ThreadTestInstance &instance) {
    ThreadReader<double> reader(instance.threadID, instance.numThreads, "./files/empty.bin");

    std::span<const double> chunk = reader.readChunk();

    CHECK(reader.getFileOpened());
    CHECK(reader.getNumItemsInFile() == 0);
    CHECK(reader.getFileSize() == 0);
    CHECK(reader.getChunkSize() == 0);
    CHECK(reader.getFirstItemOffset() == 0);
    CHECK(reader.getFirstByteOffset() == 0);

    instance.synchronize();
    reader.close();
}

/* checkReadChunk
 *
 * This function verifies that a chunk read by a thread matches the expected values for that thread's portion of the
 * file. It calculates the start and stop indices for the thread and checks that the chunk size and contents are
 * correct.
 */
template <typename Type>
void checkReadChunk(const std::vector<Type> &chunk, const std::vector<Type> &expectedValues, int threadID,
                    int numThreads, int itemCount) {
    long start = -1;
    long stop = -1;

    getChunkStartStopValues(threadID, numThreads, itemCount, start, stop);

    REQUIRE(start >= 0);
    REQUIRE(stop >= start);
    REQUIRE(static_cast<std::size_t>(stop) <= expectedValues.size());

    CHECK(chunk.size() == static_cast<std::size_t>(stop - start));

    for (std::size_t i = 0; i < chunk.size(); ++i) {
        const long expectedIndex = static_cast<std::size_t>(start) + i;

        CAPTURE(threadID);
        CAPTURE(numThreads);
        CAPTURE(itemCount);
        CAPTURE(expectedIndex);
        CHECK(static_cast<std::size_t>(expectedIndex) < expectedValues.size());

        CHECK(approximatelyEqual(chunk[i], expectedValues[expectedIndex]));
    }
}

/**
 * testReadingFileInFullMinimalConstructor
 *
 * This function tests reading a file using the minimal constructor of ThreadReader.
 * It opens the file, reads a chunk, and checks that the chunk matches the expected values.
 */
template <typename Type>
void testReadingFileInFullMinimalConstructor(const ThreadTestInstance &instance, const std::string &fileName,
                                             const std::vector<Type> &expectedValues, int itemCount) {
    ThreadReader<Type> reader(instance.threadID, instance.numThreads);
    reader.open(fileName, O_RDONLY);
    std::vector<Type> chunk = reader.readChunk();
    checkReadChunk(chunk, expectedValues, instance.threadID, instance.numThreads, itemCount);
    instance.synchronize();
    reader.close();
}

/**
 * testReadingFileInFullFileBasedConstructor
 *
 * This function tests reading a file using the full file-based constructor of ThreadReader.
 * It opens the file, reads a chunk, and checks that the chunk matches the expected values.
 */
template <typename Type>
void testReadingFileInFullFileBasedConstructor(const ThreadTestInstance &instance, const std::string &fileName,
                                               const std::vector<Type> &expectedValues, int itemCount) {
    ThreadReader<Type> reader(instance.threadID, instance.numThreads, fileName);
    std::vector<Type> chunk = reader.readChunk();
    checkReadChunk(chunk, expectedValues, instance.threadID, instance.numThreads, itemCount);
    instance.synchronize();
    reader.close();
}

/**
 * makeChunkToWrite
 *
 * This function generates a chunk of data to be written by a thread.
 * It calculates the start and stop indices for the thread and fills the chunk with values based on the thread ID and
 * item count. It is used by the testWritingFileInFullMinimalConstructor function and the
 * testWritingFileInFullFileBasedConstructor function to generate the appropriate data chunks for each thread based on
 * its ID and the total number of items.
 */
template <typename Type>
std::vector<Type> makeChunkToWrite(int threadID, int numThreads, int itemCount, long &start, long &stop) {
    getChunkStartStopValues(threadID, numThreads, itemCount, start, stop);
    std::vector<Type> chunk(static_cast<std::size_t>(stop - start));
    for (std::size_t i = 0; i < chunk.size(); ++i) {
        chunk[i] = static_cast<Type>(i + numThreads);
    }
    return chunk;
}

/**
 * checkWrittenFileByReadingBack
 *
 * This function verifies that the data written by a thread to a file matches the expected chunk.
 * It synchronizes all threads before reading to ensure that writing is complete.
 * It reads the chunk from the file and checks that the size and contents match the expected values.
 */
template <typename Type>
void checkWrittenFileByReadingBack(const ThreadTestInstance &instance, const std::string &fileName,
                                   const std::vector<Type> &expectedChunk) {
    // Every worker must finish writing/closing before anyone reads.
    instance.synchronize();
    ThreadReader<Type> reader(instance.threadID, instance.numThreads, fileName);
    std::vector<Type> valuesRead = reader.readChunk();
    CHECK(valuesRead.size() == expectedChunk.size());
    for (std::size_t i = 0; i < valuesRead.size(); ++i) {
        CHECK(i < expectedChunk.size());
        CHECK(valuesRead[i] == expectedChunk[i]);
    }
    // Do not close a shared mapping while another worker may be reading.
    instance.synchronize();
    reader.close();
}

/**
 * testWritingFileInFullMinimalConstructor
 *
 * This function tests writing a file using the minimal constructor of ThreadWriter.
 * It generates a chunk of data for the thread, writes it to the file, and verifies the file's contents.
 */
template <typename Type>
void testWritingFileInFullMinimalConstructor(const ThreadTestInstance &instance, const std::string &fileName,
                                             int itemCount) {
    long start = -1;
    long stop = -1;
    std::vector<Type> chunk = makeChunkToWrite<Type>(instance.threadID, instance.numThreads, itemCount, start, stop);
    ThreadWriter<Type> writer(instance.threadID, instance.numThreads, static_cast<long>(itemCount * sizeof(Type)));
    writer.open(fileName, O_RDWR | O_CREAT);
    writer.writeChunk(chunk);
    CHECK(writer.getNumItemsInFile() == itemCount);
    CHECK(writer.getFileSize() == static_cast<long>(itemCount * sizeof(Type)));
    CHECK(writer.getChunkSize() == static_cast<long>(chunk.size()));
    CHECK(writer.getFirstItemOffset() == start);
    CHECK(writer.getFirstByteOffset() == start * static_cast<long>(sizeof(Type)));

    instance.synchronize();
    writer.close();
    checkWrittenFileByReadingBack(instance, fileName, chunk);
}

/**
 * testWritingFileInFullFileBasedConstructor
 *
 * This function tests writing a file using the full file-based constructor of ThreadWriter.
 * It generates a chunk of data for the thread, writes it to the file, and verifies the file's contents.
 */
template <typename Type>
void testWritingFileInFullFileBasedConstructor(const ThreadTestInstance &instance, const std::string &fileName,
                                               int itemCount) {
    long start = -1;
    long stop = -1;
    std::vector<Type> chunk = makeChunkToWrite<Type>(instance.threadID, instance.numThreads, itemCount, start, stop);
    ThreadWriter<Type> writer(instance.threadID, instance.numThreads, static_cast<long>(itemCount * sizeof(Type)),
                              fileName);
    writer.writeChunk(chunk);
    CHECK(writer.getNumItemsInFile() == itemCount);
    CHECK(writer.getFileSize() == static_cast<long>(itemCount * sizeof(Type)));
    CHECK(writer.getChunkSize() == static_cast<long>(chunk.size()));
    CHECK(writer.getFirstItemOffset() == start);
    CHECK(writer.getFirstByteOffset() == start * static_cast<long>(sizeof(Type)));

    instance.synchronize();
    writer.close();
    checkWrittenFileByReadingBack(instance, fileName, chunk);
}

/**
 * runFileContentTests
 *
 * This function runs a series of tests for reading and writing file content.
 * It tests reading using both the full minimal and full file-based constructors,
 * and writing using both the full minimal and full file-based constructors.
 */
template <typename Type>
void runFileContentTests(const ThreadTestInstance &instance, const std::string &inputFile,
                         const std::vector<Type> &expectedValues, int itemCount) {

    testReadingFileInFullMinimalConstructor<Type>(instance, inputFile, expectedValues, itemCount);
    instance.synchronize();

    testReadingFileInFullFileBasedConstructor<Type>(instance, inputFile, expectedValues, itemCount);
    instance.synchronize();

    const std::string outputFile =
        "./files/test-writer-output-" + std::to_string(itemCount) + getTypeFileName<Type>() + ".bin";

    prepareOutputFile(instance, outputFile);

    testWritingFileInFullMinimalConstructor<Type>(instance, outputFile, itemCount);
    instance.synchronize();

    prepareOutputFile(instance, outputFile);

    testWritingFileInFullFileBasedConstructor<Type>(instance, outputFile, itemCount);
    instance.synchronize();
}

/*
 * Run the general ThreadReader and ThreadWriter tests.
 * @param instance The thread-specific information used for testing.
 */
void runGeneralTests(const ThreadTestInstance &instance) {

    // Test reader constructors and getters.
    testGetterFunctionsForReadersMinimalConstructor(instance);
    instance.synchronize();

    testGetterFunctionsForReadersFileBasedConstructor(instance);
    instance.synchronize();

    // Test writer constructors and getters.
    testGetterFunctionsForWritersMinimalConstructor(instance);
    instance.synchronize();

    testGetterFunctionsForWritersFileBasedConstructor(instance);
    instance.synchronize();

    // Test reader open and close operations.
    testReaderOpenCloseMinimalConstructor(instance);
    instance.synchronize();

    testReaderOpenCloseFileBasedConstructor(instance);
    instance.synchronize();

    // Test writer open and close operations.
    testWriterOpenCloseUsingMinimalConstructor(instance);
    instance.synchronize();

    testWriterOpenCloseUsingFileBasedConstructor(instance);
    instance.synchronize();

    // Test reader getters after reading a file.
    testGettersForVariablesKnownAfterReadingUsingMinimalConstructor(instance);
    instance.synchronize();

    testGettersForVariablesKnownAfterReadingUsingFileBasedConstructor(instance);
    instance.synchronize();

    // Test reading empty files.
    testReadingAnEmptyFileUsingMinimalConstructor(instance);
    instance.synchronize();

    testReadingAnEmptyFileUsingFileBasedConstructor(instance);
    instance.synchronize();
}


/*
 * Run all OpenMP file content tests using different thread counts,
 * item counts, and data types.
 */
TEST_CASE_TEMPLATE("OpenMP file content tests", Type, double, int, char) {
    omp_set_dynamic(0);

    for (const int numThreads : threadCounts) {
        for (const int itemCount : itemCounts) {

            if (numThreads > itemCount) {
                continue; // Skip cases where the number of threads exceeds the item count.
            }

            // Load expected values before starting the parallel region.
            const std::string fileName = "./files/" + std::to_string(itemCount) + getTypeFileName<Type>() + ".bin";
            std::ifstream inputFile(fileName, std::ios::binary);
            if (!inputFile.is_open()) {
                throw std::runtime_error("Could not open fixture: " + fileName);
            }
            std::vector<Type> expectedValues;
            Type value;
            for (int i = 0; i < itemCount; ++i) {
                inputFile.read(reinterpret_cast<char *>(&value), sizeof(Type));
                expectedValues.push_back(value);
            }

            std::cout << "Expected value size: " << expectedValues.size() << " Item count: " << itemCount << std::endl;

            REQUIRE(expectedValues.size() == itemCount);

            // Create the barrier using the actual OpenMP team size.
            std::unique_ptr<std::barrier<>> barrier;

#pragma omp parallel num_threads(numThreads) shared(barrier, expectedValues)
            {
#pragma omp single
                { barrier = std::make_unique<std::barrier<>>(omp_get_num_threads()); }

                ThreadTestInstance instance = setupOpenMPThreadTestInstance(*barrier);

                runFileContentTests<Type>(instance, fileName, expectedValues, itemCount);
            }
        }
    }
}

/*
 * Run general ThreadReader and ThreadWriter tests using OpenMP.
 */
TEST_CASE("OpenMP general reader and writer tests") {
    omp_set_dynamic(0);

    for (const int numThreads : threadCounts) {

        // The general tests use files containing five items.
        if (numThreads > 5) {
            continue;
        }

        std::unique_ptr<std::barrier<>> barrier;

        #pragma omp parallel num_threads(numThreads) shared(barrier)
        {
            #pragma omp single
            {
                barrier = std::make_unique<std::barrier<>>(
                    omp_get_num_threads());
            }

            ThreadTestInstance instance =
                setupOpenMPThreadTestInstance(*barrier);

            runGeneralTests(instance);
        }
    }
}

/**
 * @brief Run general ThreadReader and ThreadWriter tests using std::thread.
 */
TEST_CASE("Standard C++ thread general tests") {
    for (const int numThreads : threadCounts) {
        if (numThreads > 5) {
            continue;
        }

        runWithStdThreads(numThreads, [](const ThreadTestInstance &instance) {
            runGeneralTests(instance);
        });
    }
}

/**
 * @brief Run general ThreadReader and ThreadWriter tests using POSIX threads.
 */
TEST_CASE("POSIX thread general tests") {
    for (const int numThreads : threadCounts) {
        if (numThreads > 5) {
            continue;
        }

        runWithPthreads(numThreads, [](const ThreadTestInstance &instance) {
            runGeneralTests(instance);
        });
    }
}

/**
 * @brief Load expected values from a binary test fixture.
 * @param fileName The path to the binary fixture.
 * @param itemCount The expected number of elements.
 * @return The values read from the fixture.
 */
template <typename Type>
std::vector<Type> loadExpectedValues(
    const std::string &fileName,
    int itemCount) {

    std::ifstream inputFile(fileName, std::ios::binary);

    if (!inputFile.is_open()) {
        throw std::runtime_error("Could not open fixture: " + fileName);
    }

    std::vector<Type> expectedValues(itemCount);

    inputFile.read(
        reinterpret_cast<char *>(expectedValues.data()),
        static_cast<std::streamsize>(itemCount * sizeof(Type)));

    if (inputFile.gcount() !=
        static_cast<std::streamsize>(itemCount * sizeof(Type))) {
        throw std::runtime_error("Incomplete fixture: " + fileName);
    }

    return expectedValues;
}

/**
 * @brief Execute file-content tests for a given data type and framework.
 * @param numThreads The number of participating threads.
 * @param itemCount The number of items in the test file.
 * @param runFramework The framework-specific thread execution function.
 */
template <typename Type>
void runFileContentTestsForFramework(
    int numThreads,
    int itemCount,
    const std::function<void(
        const std::function<void(const ThreadTestInstance &)> &)> &runFramework) {

    const std::string fileName =
        "./files/" + std::to_string(itemCount) +
        getTypeFileName<Type>() + ".bin";

    const auto expectedValues =
        loadExpectedValues<Type>(fileName, itemCount);

    runFramework([&](const ThreadTestInstance &instance) {
        runFileContentTests<Type>(
            instance, fileName, expectedValues, itemCount);
    });
}

template <typename Type>
void runFileContentTestsForFramework(
    int numThreads,
    int itemCount,
    const std::function<void(
        int,
        const std::function<void(const ThreadTestInstance &)> &)> &runFramework) {

    const std::string fileName =
        "./files/" + std::to_string(itemCount) +
        getTypeFileName<Type>() + ".bin";

    const auto expectedValues =
        loadExpectedValues<Type>(fileName, itemCount);

    runFramework(numThreads, [&](const ThreadTestInstance &instance) {
        runFileContentTests<Type>(
            instance, fileName, expectedValues, itemCount);
    });
}


/**
 * @brief Run all file-content tests using standard C++ threads.
 */
TEST_CASE_TEMPLATE("Standard C++ thread file content tests", Type, double, int, char) {
    for (const int numThreads : threadCounts) {
        for (const int itemCount : itemCounts) {
            if (numThreads > itemCount) {
                continue;
            }

            runFileContentTestsForFramework<Type>(
                numThreads,
                itemCount,
                runWithStdThreads);
        }
    }
}

/**
 * @brief Run all file-content tests using POSIX threads.
 */
TEST_CASE_TEMPLATE("POSIX thread file content tests", Type, double, int, char) {
    for (const int numThreads : threadCounts) {
        for (const int itemCount : itemCounts) {
            if (numThreads > itemCount) {
                continue;
            }

            runFileContentTestsForFramework<Type>(
                numThreads,
                itemCount,
                runWithPthreads);
        }
    }
}