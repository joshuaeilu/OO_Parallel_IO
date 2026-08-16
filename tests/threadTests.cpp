/* threadTests.cpp contains tests for reading and writing files using threads.
 *
 * @author: Joshua Eilu for Joel C. Adams,
 *        Calvin University, Summer 2026
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../OO_IO/include/ThreadsIO.h"

#include <omp.h>

#include <cstdio>
#include <filesystem>

struct ThreadTestInstance {
    int threadID = 0;
    int numThreads = 1;

    void synchronize() const {
#pragma omp barrier
    }
};

ThreadTestInstance setupThreadTestInstance() {
    ThreadTestInstance instance;

    instance.threadID = omp_get_thread_num();
    instance.numThreads = omp_get_num_threads();

    return instance;
}

bool approximatelyEqual(double v1, double v2) {
    const double THRESHOLD = 0.0000000000001;

    return std::abs(v2 - v1) < THRESHOLD;
}

/*
 * Converts each C++ datatype into the word used
 * in the corresponding test filename.
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

const int threadCounts[] = {1, 2, 4, 8};
const int itemCounts[] = {1, 3, 4, 8, 9};

TEST_SUITE("File Tests") {

    TEST_CASE("getter tests for instance variables known before reading") {

        omp_set_dynamic(0); // Disable dynamic adjustment of the number of threads
        SUBCASE("using minimal constructor") {

            for (int NUM_THREADS : threadCounts) {
                CAPTURE(NUM_THREADS);
#pragma omp parallel num_threads(NUM_THREADS)
                {
                    ThreadTestInstance instance = setupThreadTestInstance();

                    ThreadReader<double> reader(instance.threadID, instance.numThreads);

                    CAPTURE(instance.threadID);

                    CHECK(reader.getID() == instance.threadID);

                    CHECK(reader.getNumPEs() == instance.numThreads);

                    CHECK(reader.getFileName() == "");

                    CHECK(reader.getItemSize() == sizeof(double));

                    CHECK_FALSE(reader.getFileOpened());
                }
            }
        }
        SUBCASE("using file-based constructor") {

            for (int NUM_THREADS : threadCounts) {
                CAPTURE(NUM_THREADS);
#pragma omp parallel num_threads(NUM_THREADS)
                {
                    ThreadTestInstance instance = setupThreadTestInstance();

                    ThreadReader<double> reader(instance.threadID, instance.numThreads, "./files/5doubles.bin");

                    CAPTURE(instance.threadID);

                    CHECK(reader.getID() == instance.threadID);

                    CHECK(reader.getNumPEs() == instance.numThreads);

                    CHECK(reader.getFileName() == "./files/5doubles.bin");

                    CHECK(reader.getItemSize() == sizeof(double));

                    CHECK(reader.getFileOpened());

                    reader.close();

                    CHECK_FALSE(reader.getFileOpened());
                }
            }
        }
    }

    TEST_CASE("getter tests for instance variables known before writing") {

        omp_set_dynamic(0); // Disable dynamic adjustment of the number of threads
        SUBCASE("using minimal constructor") {

            for (int NUM_THREADS : threadCounts) {
                CAPTURE(NUM_THREADS);
#pragma omp parallel num_threads(NUM_THREADS)
                {
                    long fileSize = 5L * static_cast<long>(sizeof(double));
                    ThreadTestInstance instance = setupThreadTestInstance();
                    ThreadWriter<double> writer(instance.threadID, instance.numThreads, fileSize);

                    CHECK(writer.getID() == instance.threadID);

                    CHECK(writer.getNumPEs() == instance.numThreads);

                    CHECK(writer.getFileName() == "");

                    CHECK(writer.getItemSize() == sizeof(double));

                    CHECK_FALSE(writer.getFileOpened());
                }
            }
        }

        SUBCASE("using file-based constructor") {

            for (int NUM_THREADS : threadCounts) {
                CAPTURE(NUM_THREADS);
#pragma omp parallel num_threads(NUM_THREADS)
                {
                    long fileSize = 5L * static_cast<long>(sizeof(double));
                    ThreadTestInstance instance = setupThreadTestInstance();
                    ThreadWriter<double> writer(instance.threadID, instance.numThreads, fileSize, "./files/5doubles.bin");
                    CHECK(writer.getID() == instance.threadID);

                    CHECK(writer.getNumPEs() == instance.numThreads);

                    CHECK(writer.getFileName() == "./files/5doubles.bin");

                    CHECK(writer.getItemSize() == sizeof(double));

                    CHECK(writer.getFileOpened());

                    writer.close();

                    CHECK_FALSE(writer.getFileOpened());
                }
            }
        }
    }

    TEST_CASE("file open/close tests") {

        omp_set_dynamic(0); // Disable dynamic adjustment of the number of threads

        SUBCASE("open/close file for reading using minimal constructor") {
            const std::string fileName = "./files/5doubles.bin";

            for (int NUM_THREADS : threadCounts) {
                CAPTURE(NUM_THREADS);

#pragma omp parallel num_threads(NUM_THREADS)
                {
                    ThreadTestInstance instance = setupThreadTestInstance();

                    CAPTURE(instance.threadID);

                    ThreadReader<double> reader(instance.threadID, instance.numThreads);

                    // Verify the initial state.
                    CHECK_FALSE(reader.getFileOpened());
                    CHECK(reader.getFileName() == "");

                    reader.open(fileName, O_RDONLY);

                    // Verify the state after open().

                    CHECK(reader.getFileName() == fileName);

                    CHECK(reader.getItemSize() == sizeof(double));

                    CHECK(reader.getFileSize() == 5L * static_cast<long>(sizeof(double)));

                    CHECK(reader.getNumItemsInFile() == 5);

                    CHECK(reader.getFileOpened());

                    /*
                     * Make sure every thread has finished checking
                     * the open state before any thread begins close().
                     */
                    instance.synchronize();

                    reader.close();

                    // Verify this object is closed.
                    CHECK_FALSE(reader.getFileOpened());
                }
            }
        }

        SUBCASE("open/close file for reading using file-based constructor") {
            const std::string fileName = "./files/5doubles.bin";

            for (int NUM_THREADS : threadCounts) {
                CAPTURE(NUM_THREADS);
#pragma omp parallel num_threads(NUM_THREADS)
                {
                    ThreadTestInstance instance = setupThreadTestInstance();

                    CAPTURE(instance.threadID);

                    ThreadReader<double> reader(instance.threadID, instance.numThreads, fileName);

                    // Verify the initial state.
                    CHECK(reader.getFileName() == fileName);
                    CHECK(reader.getItemSize() == sizeof(double));
                    CHECK(reader.getFileSize() == 5L * static_cast<long>(sizeof(double)));
                    CHECK(reader.getNumItemsInFile() == 5);
                    CHECK(reader.getFileOpened());

                    reader.close();
                    CHECK_FALSE(reader.getFileOpened());
                }
            }
        }
        SUBCASE("open/close file for writing using minimal constructor") {
            const std::string fileName = "./files/test-writer-output.bin";

            const long fileSize = 5L * static_cast<long>(sizeof(double));

            for (int NUM_THREADS : threadCounts) {
                CAPTURE(NUM_THREADS);

                std::remove(fileName.c_str());

#pragma omp parallel num_threads(NUM_THREADS)
                {
                    ThreadTestInstance instance = setupThreadTestInstance();

                    CAPTURE(instance.threadID);

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

                /*
                 * The final writer should have unmapped and closed the
                 * file, while leaving a correctly sized file on disk.
                 */
                CHECK(std::filesystem::exists(fileName));

                CHECK(static_cast<long>(std::filesystem::file_size(fileName)) == fileSize);
            }

            std::remove(fileName.c_str());
        }

        SUBCASE("open/close file for writing using file-based constructor") {
            const std::string fileName = "./files/test-writer-output.bin";

            const long fileSize = 5L * static_cast<long>(sizeof(double));

            for (int NUM_THREADS : threadCounts) {
                CAPTURE(NUM_THREADS);

                std::remove(fileName.c_str());

#pragma omp parallel num_threads(NUM_THREADS)
                {
                    ThreadTestInstance instance = setupThreadTestInstance();

                    CAPTURE(instance.threadID);

                    /*
                     * The file-based constructor should automatically
                     * call open(fileName, O_RDWR).
                     */
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

                CHECK(std::filesystem::exists(fileName));

                CHECK(static_cast<long>(std::filesystem::file_size(fileName)) == fileSize);
            }

            std::remove(fileName.c_str());
        }
    }
}

TEST_SUITE("Threaded Reading Tests") {

    TEST_CASE("getter tests for instance variables known after reading") {
        ThreadTestInstance instance = setupThreadTestInstance();

        SUBCASE("using minimal constructor") {
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

        SUBCASE("using file-based constructor") {
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
    }

    TEST_CASE("reading an empty file") {
        ThreadTestInstance instance = setupThreadTestInstance();

        SUBCASE("using minimal constructor") {
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

        SUBCASE("using file-based constructor") {
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
    }

    TEST_CASE_TEMPLATE("Reading file in full", Type, double, int, char) {
        omp_set_dynamic(0);

        SUBCASE("using minimal constructor") {
            for (int itemCount : itemCounts) {
                const std::string binaryFileName = "./files/" + std::to_string(itemCount) + getTypeFileName<Type>() + ".bin";
                const std::string textFileName = "./files/" + std::to_string(itemCount) + getTypeFileName<Type>() + ".txt";

                /*
                 * Read the expected values once, before creating
                 * the OpenMP threads.
                 */
                std::ifstream input(textFileName);
                REQUIRE(input.is_open());

                std::vector<Type> expectedValues;
                Type value;

                while (input >> value) {
                    expectedValues.push_back(value);
                }

                REQUIRE(expectedValues.size() == static_cast<std::size_t>(itemCount));

                for (int NUM_THREADS : threadCounts) {
                    if (itemCount < NUM_THREADS) {
                        std::cout << "Skipping file with " << itemCount << " items because it has fewer items than threads ("
                                  << NUM_THREADS << ").\n";

                        continue;
                    }

                    CAPTURE(itemCount);
                    CAPTURE(NUM_THREADS);
                    CAPTURE(binaryFileName);

#pragma omp parallel num_threads(NUM_THREADS)
                    {
                        ThreadTestInstance instance = setupThreadTestInstance();

                        CAPTURE(instance.threadID);

                        ThreadReader<Type> reader(instance.threadID, instance.numThreads);

                        reader.open(binaryFileName, O_RDONLY);

                        std::vector<Type> chunk = reader.readChunk();

                        long start = -1;
                        long stop = -1;

                        getChunkStartStopValues(instance.threadID, instance.numThreads, itemCount, start, stop);

                        CHECK(chunk.size() == static_cast<std::size_t>(stop - start));

                        long expectedIndex = start;

                        for (std::size_t i = 0; i < chunk.size(); ++i) {

                            CAPTURE(i);
                            CAPTURE(expectedIndex);

                            CHECK(approximatelyEqual(chunk[i], expectedValues[expectedIndex]));

                            ++expectedIndex;
                        }

                        /*
                         * Keep the mapping alive until every thread
                         * has finished examining its span.
                         */
                        instance.synchronize();

                        reader.close();
                    }
                }
            }
        }

        SUBCASE("using file-based constructor") {
            for (int itemCount : itemCounts) {
                const std::string binaryFileName = "./files/" + std::to_string(itemCount) + getTypeFileName<Type>() + ".bin";

                const std::string textFileName = "./files/" + std::to_string(itemCount) + getTypeFileName<Type>() + ".txt";

                std::ifstream input(textFileName);
                REQUIRE(input.is_open());

                std::vector<Type> expectedValues;
                Type value;

                while (input >> value) {
                    expectedValues.push_back(value);
                }

                REQUIRE(expectedValues.size() == static_cast<std::size_t>(itemCount));

                for (int NUM_THREADS : threadCounts) {
                    if (itemCount < NUM_THREADS) {
                        std::cout << "Skipping file with " << itemCount << " items because it has fewer items than threads ("
                                  << NUM_THREADS << ").\n";

                        continue;
                    }
                    CAPTURE(itemCount);
                    CAPTURE(NUM_THREADS);
                    CAPTURE(binaryFileName);

#pragma omp parallel num_threads(NUM_THREADS)
                    {
                        ThreadTestInstance instance = setupThreadTestInstance();

                        CAPTURE(instance.threadID);

                        ThreadReader<Type> reader(instance.threadID, instance.numThreads, binaryFileName);

                        std::vector<Type> chunk = reader.readChunk();

                        long start = -1;
                        long stop = -1;

                        getChunkStartStopValues(instance.threadID, instance.numThreads, itemCount, start, stop);

                        CHECK(chunk.size() == static_cast<std::size_t>(stop - start));

                        long expectedIndex = start;

                        for (std::size_t i = 0; i < chunk.size(); ++i) {

                            CAPTURE(i);
                            CAPTURE(expectedIndex);

                            CHECK(approximatelyEqual(chunk[i], expectedValues[expectedIndex]));

                            ++expectedIndex;
                        }

                        instance.synchronize();
                        reader.close();
                    }
                }
            }
        }
    }
}

TEST_SUITE("Thread Writing Tests") {

    TEST_CASE_TEMPLATE("Writing file in full", Type, double, int, char) {

        SUBCASE("using minimal constructor") {
            for (int NUM_THREADS : threadCounts) {
                CAPTURE(NUM_THREADS);
#pragma omp parallel num_threads(NUM_THREADS)
                {
                    ThreadTestInstance instance = setupThreadTestInstance();
                    for (int itemCount : itemCounts) {
                        if (itemCount < instance.numThreads) {
                            continue;
                        }

                        const std::string fileName =
                            "./files/test-minimal-" + std::to_string(itemCount) + getTypeFileName<Type>() + ".bin";

                        long start = -1, stop = -1;

                        getChunkStartStopValues(instance.threadID, instance.numThreads, itemCount, start, stop);

                        std::vector<Type> valuesToWrite;

                        for (long index = start; index < stop; ++index) {
                            valuesToWrite.push_back(static_cast<Type>(index + 1));
                        }

                        ThreadWriter<Type> writer(instance.threadID, instance.numThreads, itemCount * sizeof(Type));

                        writer.open(fileName, O_RDWR | O_CREAT);

                        writer.writeChunk(valuesToWrite);

                        CHECK(writer.getNumItemsInFile() == itemCount);

                        CHECK(writer.getFileSize() == itemCount * sizeof(Type));

                        CHECK(writer.getChunkSize() == static_cast<long>(valuesToWrite.size()));

                        CHECK(writer.getFirstItemOffset() == start);

                        CHECK(writer.getFirstByteOffset() == start * sizeof(Type));

                        writer.close();

                        instance.synchronize();

                        ThreadReader<Type> reader(instance.threadID, instance.numThreads, fileName);
                        std::vector<Type> valuesRead = reader.readChunk();
                        CHECK(valuesRead.size() == valuesToWrite.size());
                        for (std::size_t i = 0; i < valuesRead.size(); ++i) {
                            CHECK(valuesRead[i] == valuesToWrite[i]);
                        }
                        reader.close();
                        instance.synchronize();
                    }
                }
            }
        }

        SUBCASE("using file-based constructor") {

            for (int NUM_THREADS : threadCounts) {
                CAPTURE(NUM_THREADS);
#pragma omp parallel num_threads(NUM_THREADS)
                {
                    ThreadTestInstance instance = setupThreadTestInstance();
                    for (int itemCount : itemCounts) {
                        if (itemCount < instance.numThreads) {
                            continue;clear
                        }

                        const std::string fileName =
                            "./files/test-file-based-" + std::to_string(itemCount) + getTypeFileName<Type>() + ".bin";

                        long start = -1, stop = -1;

                        getChunkStartStopValues(instance.threadID, instance.numThreads, itemCount, start, stop);

                        std::vector<Type> valuesToWrite;

                        for (long index = start; index < stop; ++index) {
                            valuesToWrite.push_back(static_cast<Type>(index + 1));
                        }

                        ThreadWriter<Type> writer(instance.threadID, instance.numThreads, itemCount * sizeof(Type), fileName);

                        writer.writeChunk(valuesToWrite);

                        CHECK(writer.getNumItemsInFile() == itemCount);

                        CHECK(writer.getFileSize() == itemCount * sizeof(Type));

                        CHECK(writer.getChunkSize() == static_cast<long>(valuesToWrite.size()));

                        CHECK(writer.getFirstItemOffset() == start);

                        CHECK(writer.getFirstByteOffset() == start * sizeof(Type));

                        writer.close();

                        instance.synchronize();

                        ThreadReader<Type> reader(instance.threadID, instance.numThreads, fileName);
                        std::vector<Type> valuesRead = reader.readChunk();
                        CHECK(valuesRead.size() == valuesToWrite.size());
                        for (std::size_t i = 0; i < valuesRead.size(); ++i) {
                            CHECK(valuesRead[i] == valuesToWrite[i]);
                        }
                        reader.close();
                    }
                }
            }
        }
    }
}
