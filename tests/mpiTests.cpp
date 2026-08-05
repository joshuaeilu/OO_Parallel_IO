/* mpiTests.cpp contains tests for the MPI reader and writer.
 *
 * @author Joshua Eilu for Joel C. Adams,
 *         Calvin University, Summer 2026
 */

#define DOCTEST_CONFIG_IMPLEMENT

#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>

#include "doctest.h"

#include "../OO_IO/include/OO_MPI_IO.h"

struct MpiTestInstance {
    int rank = 0;
    int processCount = 1;

    void synchronize() const { MPI_Barrier(MPI_COMM_WORLD); }
};

MpiTestInstance setupMpiTestInstance() {
    MpiTestInstance instance;

    MPI_Comm_rank(MPI_COMM_WORLD, &instance.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &instance.processCount);

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

const int itemCounts[] = {1, 3, 4, 5, 8, 9};

TEST_SUITE("File Tests") {

    TEST_CASE("getter tests for instance variables known before reading") {
        MpiTestInstance instance = setupMpiTestInstance();

        SUBCASE("using minimal constructor") {
            MPIProcessReader<double> reader(instance.rank, instance.processCount);

            CHECK(reader.getID() == instance.rank);
            CHECK(reader.getNumPEs() == instance.processCount);
            CHECK(reader.getFileName() == "");
            CHECK(reader.getItemSize() == sizeof(double));
            CHECK(reader.getFileOpened() == false);
        }

        SUBCASE("using file-based constructor") {
            MPIProcessReader<double> reader(instance.rank, instance.processCount, "./files/5doubles.bin");

            CHECK(reader.getID() == instance.rank);
            CHECK(reader.getNumPEs() == instance.processCount);
            CHECK(reader.getFileName() == "./files/5doubles.bin");
            CHECK(reader.getItemSize() == sizeof(double));
            CHECK(reader.getFileOpened() == true);

            reader.close();

            CHECK(reader.getFileOpened() == false);
        }
    }

    TEST_CASE("getter tests for instance variables known before writing") {
        MpiTestInstance instance = setupMpiTestInstance();

        SUBCASE("using minimal constructor") {
            MPIProcessWriter<double> writer(instance.rank, instance.processCount);

            CHECK(writer.getID() == instance.rank);
            CHECK(writer.getNumPEs() == instance.processCount);
            CHECK(writer.getFileName() == "");
            CHECK(writer.getItemSize() == sizeof(double));
            CHECK(writer.getFileOpened() == false);
        }

        SUBCASE("using file-based constructor") {
            MPIProcessWriter<double> writer(instance.rank, instance.processCount, "./files/5doubles.bin");

            CHECK(writer.getID() == instance.rank);
            CHECK(writer.getNumPEs() == instance.processCount);
            CHECK(writer.getFileName() == "./files/5doubles.bin");
            CHECK(writer.getItemSize() == sizeof(double));
            CHECK(writer.getFileOpened() == true);

            writer.close();
        }
    }

    TEST_CASE("file open/close tests") {
        MpiTestInstance instance = setupMpiTestInstance();

        SUBCASE("open and close a file for reading using minimal constructor") {
            MPIProcessReader<double> reader(instance.rank, instance.processCount);

            CHECK(reader.getFileOpened() == false);

            reader.open("./files/5doubles.bin", MPI_MODE_RDONLY);
            CHECK(reader.getFileOpened() == true);

            reader.close();
            CHECK(reader.getFileOpened() == false);
        }

        SUBCASE("open and close a file for writing using minimal constructor") {
            MPIProcessWriter<double> writer(instance.rank, instance.processCount);

            CHECK(writer.getFileOpened() == false);

            writer.open("./files/5doubles.bin", MPI_MODE_WRONLY | MPI_MODE_CREATE);
            CHECK(writer.getFileOpened() == true);
            CHECK(writer.getFileSize() == 0);
            writer.close();
            CHECK(writer.getFileOpened() == false);
        }

        SUBCASE("open and close a file for reading using file-based constructor") {
            MPIProcessReader<double> reader(instance.rank, instance.processCount, "./files/5doubles.bin");

            CHECK(reader.getFileOpened() == true);
            reader.close();
            CHECK(reader.getFileOpened() == false);
        }

        SUBCASE("open and close a file for writing using file-based constructor") {
            MPIProcessWriter<double> writer(instance.rank, instance.processCount, "./files/5doubles.bin");

            CHECK(writer.getFileOpened() == true);
            writer.close();
            CHECK(writer.getFileOpened() == false);
        }
    }
}

TEST_SUITE("MPI Reading Tests") {

    TEST_CASE("getter tests for instance variables known after reading") {
        MpiTestInstance instance = setupMpiTestInstance();

        SUBCASE("using minimal constructor") {
            MPIProcessReader<double> reader(instance.rank, instance.processCount);

            reader.open("./files/5doubles.bin", MPI_MODE_RDONLY);
            std::vector<double> chunk = reader.readChunk();

            CHECK(reader.getFileOpened() == true);
            CHECK(reader.getNumItemsInFile() == 5);
            CHECK(reader.getFileSize() == 5 * sizeof(double));
            CHECK(reader.getChunkSize() == 5 / instance.processCount + (instance.rank < 5 % instance.processCount));
            CHECK(reader.getFirstItemOffset() ==
                  instance.rank * (5 / instance.processCount) + std::min(instance.rank, 5 % instance.processCount));
            CHECK(reader.getFirstByteOffset() ==
                  (instance.rank * (5 / instance.processCount) + std::min(instance.rank, 5 % instance.processCount)) *
                      sizeof(double));

            reader.close();
        }

        SUBCASE("using file-based constructor") {
            MPIProcessReader<double> reader(instance.rank, instance.processCount, "./files/5doubles.bin");

            std::vector<double> chunk = reader.readChunk();

            CHECK(reader.getFileOpened() == true);
            CHECK(reader.getNumItemsInFile() == 5);
            CHECK(reader.getFileSize() == 5 * sizeof(double));
            CHECK(reader.getChunkSize() == 5 / instance.processCount + (instance.rank < 5 % instance.processCount));
            CHECK(reader.getFirstItemOffset() ==
                  instance.rank * (5 / instance.processCount) + std::min(instance.rank, 5 % instance.processCount));
            CHECK(reader.getFirstByteOffset() ==
                  (instance.rank * (5 / instance.processCount) + std::min(instance.rank, 5 % instance.processCount)) *
                      sizeof(double));

            reader.close();
        }
    }

    TEST_CASE("reading an empty file") {
        MpiTestInstance instance = setupMpiTestInstance();

        SUBCASE("using minimal constructor") {
            MPIProcessReader<double> reader(instance.rank, instance.processCount);

            reader.open("./files/empty.bin", MPI_MODE_RDONLY);
            std::vector<double> chunk = reader.readChunk();

            CHECK(reader.getFileOpened() == true);
            CHECK(reader.getNumItemsInFile() == 0);
            CHECK(reader.getFileSize() == 0);
            CHECK(reader.getChunkSize() == 0);
            CHECK(reader.getFirstItemOffset() == 0);
            CHECK(reader.getFirstByteOffset() == 0);

            CHECK(chunk.empty());

            reader.close();
        }

        SUBCASE("using file-based constructor") {
            MPIProcessReader<double> reader(instance.rank, instance.processCount, "./files/empty.bin");

            std::vector<double> chunk = reader.readChunk();

            CHECK(reader.getFileOpened() == true);
            CHECK(reader.getNumItemsInFile() == 0);
            CHECK(reader.getFileSize() == 0);
            CHECK(reader.getChunkSize() == 0);
            CHECK(reader.getFirstItemOffset() == 0);
            CHECK(reader.getFirstByteOffset() == 0);

            CHECK(chunk.empty());

            reader.close();
        }
    }

    TEST_CASE_TEMPLATE("Reading file in full", Type, double, int, char) {
        MpiTestInstance instance = setupMpiTestInstance();

        // We are going to test using different file sizes

        SUBCASE("using minimal constructor") {
            for (int itemCount : itemCounts) {
                if (itemCount < instance.processCount) {
                    if (instance.rank == 0) {
                        std::cout << "Skipping test for file with " << itemCount << " items"
                                  << " because it is less than the number of processes (" << instance.processCount << ")."
                                  << std::endl;
                    }
                    continue;
                }
                std::string fileName = "./files/" + std::to_string(itemCount) + getTypeFileName<Type>() + ".bin";
                MPIProcessReader<Type> reader(instance.rank, instance.processCount);
                reader.open(fileName, MPI_MODE_RDONLY);
                std::vector<Type> chunk = reader.readChunk();

                const int SIZE = reader.getNumItemsInFile();
                std::vector<Type> expectedValues;
                Type dVal;
                std::ifstream fin("./files/" + std::to_string(itemCount) + getTypeFileName<Type>() + ".txt");
                CHECK(fin.is_open());
                for (int i = 0; i < SIZE; ++i) {
                    fin >> dVal;
                    expectedValues.push_back(dVal);
                }
                fin.close();

                long start = -1, stop = -1;
                getChunkStartStopValues(instance.rank, instance.processCount, SIZE, start, stop);
                CHECK(chunk.size() == static_cast<size_t>(stop - start));

                int j = start;
                for (int i = 0; i < static_cast<int>(chunk.size()); ++i) {
                    CHECK(approximatelyEqual(chunk[i], expectedValues[j]));
                    ++j;
                }

                reader.close();
            }
        }

        SUBCASE("using file-based constructor") {

            for (int itemCount : itemCounts) {
                if (itemCount < instance.processCount) {
                    if (instance.rank == 0) {
                        std::cout << "Skipping test for file with " << itemCount << " items"
                                  << " because it is less than the number of processes (" << instance.processCount << ")."
                                  << std::endl;
                    }
                    continue;
                }
                std::string fileName = "./files/" + std::to_string(itemCount) + getTypeFileName<Type>() + ".bin";
                MPIProcessReader<Type> reader(instance.rank, instance.processCount, fileName);
                std::vector<Type> chunk = reader.readChunk();

                const int SIZE = reader.getNumItemsInFile();
                std::vector<Type> expectedValues;
                Type dVal;
                std::ifstream fin("./files/" + std::to_string(itemCount) + getTypeFileName<Type>() + ".txt");
                CHECK(fin.is_open());
                for (int i = 0; i < SIZE; ++i) {
                    fin >> dVal;
                    expectedValues.push_back(dVal);
                }
                fin.close();

                long start = -1, stop = -1;
                getChunkStartStopValues(instance.rank, instance.processCount, SIZE, start, stop);
                CHECK(chunk.size() == static_cast<size_t>(stop - start));

                int j = start;
                for (int i = 0; i < static_cast<int>(chunk.size()); ++i) {
                    CHECK(approximatelyEqual(chunk[i], expectedValues[j]));
                    ++j;
                }

                reader.close();
            }
        }
    }

    TEST_CASE_TEMPLATE("Read chunk : reader correctly distributes file items among processes", Type, double, int, char) {
        MpiTestInstance instance = setupMpiTestInstance();

        for (long itemCount : itemCounts) {
            const std::string fileName = "./files/" + std::to_string(itemCount) + getTypeFileName<Type>() + ".bin";

            if (itemCount < instance.processCount) {
                if (instance.rank == 0) {
                    std::cout << "Skipping test for file with " << itemCount << " items"
                              << " because it is less than the number of processes (" << instance.processCount << ")."
                              << std::endl;
                }
                continue;
            }

            CAPTURE(instance.rank);
            CAPTURE(instance.processCount);
            CAPTURE(itemCount);
            CAPTURE(fileName);

            MPIProcessReader<Type> reader(instance.rank, instance.processCount, fileName);

            const long baseChunkSize = itemCount / instance.processCount;

            const long remainder = itemCount % instance.processCount;

            const long expectedChunkSize = baseChunkSize + (instance.rank < remainder ? 1 : 0);

            const long expectedFirstItemOffset = instance.rank * baseChunkSize + std::min<long>(instance.rank, remainder);

            const long expectedFirstByteOffset = expectedFirstItemOffset * sizeof(Type);
            reader.readChunk();

            CHECK(reader.getChunkSize() == expectedChunkSize);

            CHECK(reader.getFirstItemOffset() == expectedFirstItemOffset);

            CHECK(reader.getFirstByteOffset() == expectedFirstByteOffset);

            reader.close();
            instance.synchronize();
        }
    }

    TEST_CASE_TEMPLATE("Reading chunk plus", Type, double, int, char) {
        MpiTestInstance instance = setupMpiTestInstance();

        const unsigned extraCounts[] = {1, 2, 3, 6, 10};

        for (int itemCount : itemCounts) {
            if (itemCount < instance.processCount) {
                continue;
            }

            const std::string binaryFileName = "./files/" + std::to_string(itemCount) + getTypeFileName<Type>() + ".bin";
            const std::string textFileName = "./files/" + std::to_string(itemCount) + getTypeFileName<Type>() + ".txt";

            MPIProcessReader<Type> reader(instance.rank, instance.processCount, binaryFileName);

            for (unsigned numExtras : extraCounts) {
                CAPTURE(instance.rank);
                CAPTURE(instance.processCount);
                CAPTURE(itemCount);
                CAPTURE(numExtras);
                CAPTURE(binaryFileName);

                std::vector<Type> actualValues = reader.readChunkPlus(numExtras);

                CHECK(reader.getFileSize() == itemCount * sizeof(Type));

                CHECK(reader.getNumItemsInFile() == itemCount);

                // Load the complete expected file.
                std::ifstream input(textFileName);

                REQUIRE(input.is_open());

                std::vector<Type> expectedValues;
                Type value;

                for (int i = 0; i < itemCount; ++i) {
                    input >> value;
                    expectedValues.push_back(value);
                }

                input.close();

                REQUIRE(expectedValues.size() == static_cast<std::size_t>(itemCount));

                long start = -1, stop = -1;

                getChunkStartStopValues(instance.rank, instance.processCount, itemCount, start, stop);

                /*
                 * Every process except the last one may read
                 * additional overlapping items.
                 */
                if (instance.rank < instance.processCount - 1) {
                    stop += numExtras;
                }

                /*
                 * Do not read beyond the end of the file.
                 */
                if (stop > itemCount) {
                    stop = itemCount;
                }

                CHECK(actualValues.size() == static_cast<std::size_t>(stop - start));

                /*
                 * Verify the normal and extra values.
                 */
                long expectedIndex = start;

                for (std::size_t i = 0; i < actualValues.size(); ++i) {
                    CAPTURE(i);
                    CAPTURE(expectedIndex);

                    CHECK(approximatelyEqual(actualValues[i], expectedValues[expectedIndex]));

                    ++expectedIndex;
                }

                instance.synchronize();
            }

            reader.close();
            instance.synchronize();
        }
    }

    TEST_CASE_TEMPLATE("Read chunk plus: reader correctly calculates chunk information", Type, double, int, char) {
        MpiTestInstance instance = setupMpiTestInstance();

        const unsigned extraCounts[] = {1, 2, 3, 6, 10};

        for (long itemCount : itemCounts) {
            if (itemCount < instance.processCount) {
                continue;
            }

            const std::string fileName = "./files/" + std::to_string(itemCount) + getTypeFileName<Type>() + ".bin";

            for (unsigned numExtras : extraCounts) {
                CAPTURE(instance.rank);
                CAPTURE(instance.processCount);
                CAPTURE(itemCount);
                CAPTURE(numExtras);
                CAPTURE(fileName);

                MPIProcessReader<Type> reader(instance.rank, instance.processCount, fileName);

                /*
                 * Calculate the normal chunk distribution.
                 */
                const long baseChunkSize = itemCount / instance.processCount;

                const long remainder = itemCount % instance.processCount;

                const long normalChunkSize = baseChunkSize + (instance.rank < remainder ? 1 : 0);

                const long expectedFirstItemOffset = instance.rank * baseChunkSize + std::min<long>(instance.rank, remainder);

                // position after processor's normal chunk, before adding extras
                const long normalStop = expectedFirstItemOffset + normalChunkSize;

                /*
                 * Extend the normal stop position, but do not
                 * continue past the end of the file.
                 */
                const long expectedStop = std::min<long>(normalStop + numExtras, itemCount);

                const long expectedChunkSize = expectedStop - expectedFirstItemOffset;

                const long expectedFirstByteOffset = expectedFirstItemOffset * sizeof(Type);

                reader.readChunkPlus(numExtras);

                CHECK(reader.getChunkSize() == expectedChunkSize);

                CHECK(reader.getFirstItemOffset() == expectedFirstItemOffset);

                CHECK(reader.getFirstByteOffset() == expectedFirstByteOffset);

                reader.close();
                instance.synchronize();
            }
        }
    }
}

TEST_SUITE("MPI Writing Tests") {

    TEST_CASE_TEMPLATE("Writing file in full", Type, double, int, char) {
        MpiTestInstance instance = setupMpiTestInstance();

        SUBCASE("using minimal constructor") {
            for (int itemCount : itemCounts) {
                if (itemCount < instance.processCount) {
                    if (instance.rank == 0) {
                        std::cout << "Skipping test for file with " << itemCount << " items because it is less than the number "
                                  << "of processes (" << instance.processCount << ")." << std::endl;
                    }

                    continue;
                }

                const std::string fileName =
                    "./files/test-minimal-" + std::to_string(itemCount) + getTypeFileName<Type>() + ".bin";

                long start = -1, stop = -1;

                getChunkStartStopValues(instance.rank, instance.processCount, itemCount, start, stop);

                std::vector<Type> valuesToWrite;

                for (long index = start; index < stop; ++index) {
                    valuesToWrite.push_back(static_cast<Type>(index + 1) * 11.1);
                }

                MPIProcessWriter<Type> writer(instance.rank, instance.processCount);

                writer.open(fileName, MPI_MODE_WRONLY | MPI_MODE_CREATE);

                writer.writeChunk(valuesToWrite);

                CHECK(writer.getNumItemsInFile() == itemCount);

                CHECK(writer.getFileSize() == itemCount * sizeof(Type));

                CHECK(writer.getChunkSize() == static_cast<long>(valuesToWrite.size()));

                CHECK(writer.getFirstItemOffset() == start);

                CHECK(writer.getFirstByteOffset() == start * sizeof(Type));

                writer.close();

                instance.synchronize();

                MPIProcessReader<Type> reader(instance.rank, instance.processCount, fileName);

                std::vector<Type> valuesRead = reader.readChunk();

                CHECK(valuesRead.size() == valuesToWrite.size());

                for (std::size_t i = 0; i < valuesRead.size(); ++i) {
                    CHECK(valuesRead[i] == valuesToWrite[i]);
                }

                reader.close();

                instance.synchronize();
            }
        }

        SUBCASE("using file-based constructor") {
            for (int itemCount : itemCounts) {
                if (itemCount < instance.processCount) {
                    if (instance.rank == 0) {
                        std::cout << "Skipping test for file with " << itemCount << " items because it is less than the number "
                                  << "of processes (" << instance.processCount << ")." << std::endl;
                    }

                    continue;
                }

                const std::string fileName =
                    "./files/test-file-based-" + std::to_string(itemCount) + getTypeFileName<Type>() + ".bin";

                long start = -1, stop = -1;

                getChunkStartStopValues(instance.rank, instance.processCount, itemCount, start, stop);

                std::vector<Type> valuesToWrite;

                for (long index = start; index < stop; ++index) {
                    valuesToWrite.push_back(static_cast<Type>(index + 1));
                }

                MPIProcessWriter<Type> writer(instance.rank, instance.processCount, fileName);

                writer.writeChunk(valuesToWrite);

                CHECK(writer.getNumItemsInFile() == itemCount);

                CHECK(writer.getFileSize() == itemCount * sizeof(Type));

                CHECK(writer.getChunkSize() == static_cast<long>(valuesToWrite.size()));

                CHECK(writer.getFirstItemOffset() == start);

                CHECK(writer.getFirstByteOffset() == start * sizeof(Type));

                writer.close();

                instance.synchronize();

                MPIProcessReader<Type> reader(instance.rank, instance.processCount, fileName);

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

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    doctest::Context context;
    context.applyCommandLine(argc, argv);

    int testResult = context.run();

    MPI_Finalize();

    return testResult;
}