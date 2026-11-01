/*
 * cudaTests.cpp contains tests for reading and writing files using
 * CUDA and NVIDIA GPUDirect Storage (GDS).
 *
 * @author: Joshua Eilu for Joel C. Adams,
 *          Calvin University, Summer 2026
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../OO_IO/include/CUDAIO.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

// ------------------------------------------------------------
// CUDA kernel used by tests
// ------------------------------------------------------------

__global__ void doubleValues(double *data, size_t numItems) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numItems) {
        data[index] *= 2.0;
    }
}

// -----------------------------------------------------------------------------
// Test constants
// -----------------------------------------------------------------------------

namespace {

const std::string FIVE_DOUBLES_FILE = "./files/5doubles.bin";

const std::string EMPTY_FILE = "./files/empty.bin";

const std::string ONE_MILLION_DOUBLES_FILE = "./files/1m-doubles.bin";

const std::string TWO_BILLION_DOUBLES_FILE = "./files/2b-doubles.bin";

constexpr std::size_t ONE_MILLION = 1'000'000ULL;

// -----------------------------------------------------------------------------
// Helper: read expected values using ordinary CPU I/O
// -----------------------------------------------------------------------------

template <class ItemType>
std::vector<ItemType> readExpectedValues(const std::string &fileName, std::size_t numItems) {
    std::vector<ItemType> values(numItems);

    std::ifstream input(fileName, std::ios::binary);

    REQUIRE(input.is_open());

    const std::size_t numBytes = numItems * sizeof(ItemType);

    input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(numBytes));

    REQUIRE(input.gcount() == static_cast<std::streamsize>(numBytes));

    return values;
}

// -----------------------------------------------------------------------------
// Helper: copy GPU data back to CPU for test verification
// -----------------------------------------------------------------------------

template <class ItemType>
std::vector<ItemType> copyGPUDataToHostMemory(const ItemType *devicePtr, std::size_t numItems) {
    std::vector<ItemType> hostBuffer(numItems);

    const std::size_t numBytes = numItems * sizeof(ItemType);

    cudaError_t cudaStatus = cudaMemcpy(hostBuffer.data(), devicePtr, numBytes, cudaMemcpyDeviceToHost);

    REQUIRE(cudaStatus == cudaSuccess);

    return hostBuffer;
}

// -----------------------------------------------------------------------------
// Helper: verify two vectors contain the same values
// -----------------------------------------------------------------------------

template <class ItemType>
void checkValuesEqual(const std::vector<ItemType> &actual, const std::vector<ItemType> &expected) {
    REQUIRE(actual.size() == expected.size());


    bool valuesMatch = true;
    std::size_t mismatchIndex = 0;

    for (std::size_t i = 0; i < actual.size(); ++i) {

        if (actual[i] != expected[i]) {
            valuesMatch = false;
            mismatchIndex = i;
            break;
        }
    }

    if (!valuesMatch) {
        CAPTURE(mismatchIndex);
        CAPTURE(actual[mismatchIndex]);
        CAPTURE(expected[mismatchIndex]);
    }

    CHECK(valuesMatch);
}

} // namespace

// =============================================================================
// CUDA Environment Tests
// =============================================================================

TEST_SUITE("CUDA Environment Tests") {

    TEST_CASE("CUDA device is available") {

        int deviceCount = 0;

        cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

        REQUIRE(cudaStatus == cudaSuccess);

        CHECK(deviceCount > 0);
    }
}

// =============================================================================
// CUDA File Tests
// =============================================================================

TEST_SUITE("CUDA File Tests") {

    TEST_CASE("minimal CUDAReader constructor") {

        gpuReader<double> reader;

        CHECK(reader.getID() == 0);
        CHECK(reader.getNumPEs() == 1);

        CHECK(reader.getFileName() == "");

        CHECK(reader.getItemSize() == sizeof(double));

        CHECK_FALSE(reader.getFileOpened());
    }

    TEST_CASE("file-based CUDAReader constructor") {

        gpuReader<double> reader(FIVE_DOUBLES_FILE);

        CHECK(reader.getID() == 0);
        CHECK(reader.getNumPEs() == 1);

        CHECK(reader.getFileName() == FIVE_DOUBLES_FILE);

        CHECK(reader.getItemSize() == sizeof(double));

        CHECK(reader.getFileSize() == 5L * static_cast<long>(sizeof(double)));

        CHECK(reader.getNumItemsInFile() == 5);

        CHECK(reader.getFileOpened());

        reader.close();

        CHECK_FALSE(reader.getFileOpened());
    }

    TEST_CASE("open and close reader using minimal constructor") {

        gpuReader<double> reader;

        CHECK_FALSE(reader.getFileOpened());

        reader.open(FIVE_DOUBLES_FILE, O_RDONLY | O_DIRECT);

        CHECK(reader.getFileOpened());

        CHECK(reader.getFileName() == FIVE_DOUBLES_FILE);

        CHECK(reader.getItemSize() == sizeof(double));

        CHECK(reader.getFileSize() == 5L * static_cast<long>(sizeof(double)));

        CHECK(reader.getNumItemsInFile() == 5);

        reader.close();

        CHECK_FALSE(reader.getFileOpened());
    }

    TEST_CASE("invalid write-only access mode is rejected") {

        gpuReader<double> reader;

        CHECK_THROWS_AS(reader.open(FIVE_DOUBLES_FILE, O_WRONLY | O_DIRECT), std::invalid_argument);
    }

    TEST_CASE("reading without opening a file throws") {

        gpuReader<double> reader;
        int testIndex = 0;

        CHECK_THROWS_AS(reader.readChunk(testIndex), std::runtime_error);
    }
}

// =============================================================================
// CUDA Reading Tests
// =============================================================================

TEST_SUITE("CUDA Reading Tests") {

    TEST_CASE("reading an empty file using minimal constructor") {

        gpuReader<double> reader;

        reader.open(EMPTY_FILE, O_RDONLY | O_DIRECT);

        CHECK(reader.getFileOpened());

        CHECK(reader.getNumItemsInFile() == 0);

        CHECK(reader.getFileSize() == 0);
        reader.setDeviceBufferSize(1024 * sizeof(double)); // Set GPU buffer size to hold 1024 doubles

        double *gpuData = reader.readChunk(0);

        CHECK(gpuData == nullptr);

        reader.close();

        CHECK_FALSE(reader.getFileOpened());
    }

    TEST_CASE("reading an empty file using file-based constructor") {

        gpuReader<double> reader(EMPTY_FILE);

        CHECK(reader.getFileOpened());

        CHECK(reader.getNumItemsInFile() == 0);

        CHECK(reader.getFileSize() == 0);
        reader.setDeviceBufferSize(1024 * sizeof(double)); // Set GPU buffer size to hold 1024 doubles

        double *gpuData = reader.readChunk(0);

        CHECK(gpuData == nullptr);

        reader.close();

        CHECK_FALSE(reader.getFileOpened());
    }

    TEST_CASE("reading 1 million doubles that can fit into GPU") {

        const std::size_t numItems = ONE_MILLION;

        const std::size_t numBytes = numItems * sizeof(double);

        // Read expected values using ordinary CPU I/O.
        std::vector<double> expected = readExpectedValues<double>(ONE_MILLION_DOUBLES_FILE, numItems);

        SUBCASE("Using file-based constructor") {

            // Read the same file directly into GPU memory using GDS.
            gpuReader<double> reader(ONE_MILLION_DOUBLES_FILE);

            CHECK(reader.getFileOpened());

            CHECK(reader.getNumItemsInFile() == numItems);

            CHECK(reader.getFileSize() == static_cast<long>(numBytes));

            // set buffersize to the number of bytes we want to read
            reader.setDeviceBufferSize(numBytes);

            CHECK(reader.getNumItemsInFile() == numItems);

            double *gpuData = reader.readChunk(0);

            REQUIRE(gpuData != nullptr);

            // Verify that readChunk() really returned device memory.
            cudaPointerAttributes attributes{};
            cudaError_t cudaStatus = cudaPointerGetAttributes(&attributes, gpuData);
            REQUIRE(cudaStatus == cudaSuccess);
            CHECK(attributes.type == cudaMemoryTypeDevice);

            // Copy GPU data back to CPU ONLY for verification.
            std::vector<double> actual = copyGPUDataToHostMemory(gpuData, numItems);
            checkValuesEqual(actual, expected);

            reader.close();

            CHECK_FALSE(reader.getFileOpened());
        }

        SUBCASE("Using the minimal constructor") {
            gpuReader<double> reader;

            reader.open(ONE_MILLION_DOUBLES_FILE, O_RDONLY | O_DIRECT);

            CHECK(reader.getFileOpened());

            CHECK(reader.getNumItemsInFile() == numItems);

            CHECK(reader.getFileSize() == static_cast<long>(numBytes));

            reader.setDeviceBufferSize(numBytes);
            CHECK(reader.getNumItemsInFile() == numItems);

            double *gpuData = reader.readChunk(0);

            REQUIRE(gpuData != nullptr);

            // Verify that readChunk() really returned device memory.
            cudaPointerAttributes attributes{};
            cudaError_t cudaStatus = cudaPointerGetAttributes(&attributes, gpuData);
            REQUIRE(cudaStatus == cudaSuccess);
            CHECK(attributes.type == cudaMemoryTypeDevice);

            // Copy GPU data back to CPU ONLY for verification.
            std::vector<double> actual = copyGPUDataToHostMemory(gpuData, numItems);

            // Verify all values.
            checkValuesEqual(actual, expected);

            reader.close();

            CHECK_FALSE(reader.getFileOpened());
        }
    }

    TEST_CASE("reading 2 billion doubles that may not fit into GPU") {

        SUBCASE("allocating a buffer size that exceeds GPU memory") {
            gpuReader<double> reader(TWO_BILLION_DOUBLES_FILE);

            const std::size_t numItems = 2'000'000'000;

            CHECK(reader.getFileOpened());

            CHECK(reader.getNumItemsInFile() == numItems);
            reader.setDeviceBufferSize(static_cast<std::size_t>(numItems * sizeof(double))); // 8 GB

            CHECK_THROWS_AS(reader.readChunk(0), std::runtime_error);

            reader.close();

            CHECK_FALSE(reader.getFileOpened());
        }

        SUBCASE("using readChunk() to read a large file in chunks") {

            gpuReader<double> reader(TWO_BILLION_DOUBLES_FILE);
            const std::size_t numItems = 2'000'000'000;

            REQUIRE(reader.getFileOpened());
            CHECK(reader.getNumItemsInFile() == numItems);
            CHECK(reader.getFileSize() == static_cast<long>(numItems * sizeof(double)));

            reader.setDeviceBufferSize(2 * 1024 * 1024 * 1024ULL); // 2 GB buffer

            std::size_t totalItemsRead = 0;

            std::vector<double> valuesToCompare = readExpectedValues<double>(TWO_BILLION_DOUBLES_FILE, numItems);

            for (std::size_t chunkIndex = 0; chunkIndex < reader.getNumChunks(); ++chunkIndex) {

                
                // Read the chunk into GPU memory.
                double *gpuData = reader.readChunk(chunkIndex);
                const std::size_t currentChunkItems = reader.getChunkSize() / sizeof(double);

                REQUIRE(gpuData != nullptr);

                // Copy the chunk back to CPU using existing helper.
                std::vector<double> actual = copyGPUDataToHostMemory(gpuData, currentChunkItems);

                // Generate expected values for this chunk.
                std::vector<double> expected(currentChunkItems);
                for (std::size_t i = 0; i < currentChunkItems; ++i) {
                    expected[i] = valuesToCompare[totalItemsRead + i];
                } 


                // Verify using existing helper.
                checkValuesEqual(actual, expected);

                totalItemsRead += currentChunkItems;
            }

            CHECK(totalItemsRead == numItems);

            reader.close();
            CHECK_FALSE(reader.getFileOpened());
        }
    }
}
// =============================================================================
// CUDA Writing Tests
// =============================================================================

// TEST_SUITE("CUDA Writing Tests") {

//     // -------------------------------------------------------------------------
//     // Constructor tests
//     // -------------------------------------------------------------------------

//     TEST_CASE("minimal CUDAWriter constructor") {

//         const long fileSize = 4096;

//         CUDAWriter<double> writer(fileSize);

//         CHECK(writer.getID() == 0);
//         CHECK(writer.getNumPEs() == 1);

//         CHECK(writer.getFileName() == "");

//         CHECK(writer.getItemSize() == sizeof(double));

//         CHECK(writer.getFileSize() == fileSize);

//         CHECK(writer.getNumItemsInFile() == static_cast<std::size_t>(fileSize) / sizeof(double));

//         CHECK_FALSE(writer.getFileOpened());
//     }

//     TEST_CASE("CUDAWriter rejects negative file size") {

//         CHECK_THROWS_AS(CUDAWriter<double>(-1), std::invalid_argument);
//     }

//     TEST_CASE("CUDAWriter rejects file size not divisible by ItemType") {

//         CHECK_THROWS_AS(CUDAWriter<double>(4097), std::invalid_argument);
//     }

//     TEST_CASE("file-based CUDAWriter constructor") {

//         const std::string fileName = "./files/cuda-writer-constructor.bin";
//         std::remove(fileName.c_str());

//         const long fileSize = 4096;

//         CUDAWriter<double> writer(fileSize, fileName);

//         CHECK(writer.getFileOpened());

//         CHECK(writer.getFileName() == fileName);

//         CHECK(writer.getFileSize() == fileSize);

//         CHECK(writer.getNumItemsInFile() == static_cast<std::size_t>(fileSize) / sizeof(double));

//         writer.close();
//         std::remove(fileName.c_str());

//         CHECK_FALSE(writer.getFileOpened());
//     }

//     TEST_CASE("open and close CUDAWriter using minimal constructor") {

//         const std::string fileName = "./files/cuda-writer-open-close.bin";

//         std::remove(fileName.c_str());

//         CUDAWriter<double> writer(4096);

//         CHECK_FALSE(writer.getFileOpened());

//         writer.open(fileName, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT);

//         CHECK(writer.getFileOpened());

//         CHECK(writer.getFileName() == fileName);

//         CHECK(writer.getFileSize() == 4096);

//         writer.close();

//         CHECK_FALSE(writer.getFileOpened());

//         std::remove(fileName.c_str());
//     }

//     // -------------------------------------------------------------------------
//     // Writing without opening
//     // -------------------------------------------------------------------------

//     TEST_CASE("writeToFile throws when file is not open") {

//         CUDAWriter<double> writer(4096);

//         double *gpuData = nullptr;

//         CHECK_FALSE(writer.getFileOpened());

//         CHECK_THROWS_AS(writer.writeToFile(gpuData, 4096 / sizeof(double)), std::runtime_error);
//     }
// }