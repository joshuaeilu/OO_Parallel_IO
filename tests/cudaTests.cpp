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

// -----------------------------------------------------------------------------
// Test constants
// -----------------------------------------------------------------------------

namespace {

const std::string FIVE_DOUBLES_FILE = "./files/5doubles.bin";

const std::string EMPTY_FILE = "./files/empty.bin";

const std::string ONE_MILLION_DOUBLES_FILE = "./files/1m-doubles.bin";

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
std::vector<ItemType> copyGPUDataToDeviceMemory(const ItemType *devicePtr, std::size_t numItems) {
    std::vector<ItemType> hostBuffer(numItems);

    const std::size_t numBytes = numItems * sizeof(ItemType);

    cudaError_t cudaStatus =
        cudaMemcpy(hostBuffer.data(), devicePtr, numBytes, cudaMemcpyDeviceToHost);

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

        CUDAReader<double> reader;

        CHECK(reader.getID() == 0);
        CHECK(reader.getNumPEs() == 1);

        CHECK(reader.getFileName() == "");

        CHECK(reader.getItemSize() == sizeof(double));

        CHECK_FALSE(reader.getFileOpened());
    }

    TEST_CASE("file-based CUDAReader constructor") {

        CUDAReader<double> reader(FIVE_DOUBLES_FILE);

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

        CUDAReader<double> reader;

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

        CUDAReader<double> reader;

        CHECK_THROWS_AS(reader.open(FIVE_DOUBLES_FILE, O_WRONLY | O_DIRECT), std::invalid_argument);
    }

    TEST_CASE("reading without opening a file throws") {

        CUDAReader<double> reader;

        CHECK_THROWS_AS(reader.readToGPU(), std::runtime_error);
    }
}

// =============================================================================
// CUDA Reading Tests
// =============================================================================

TEST_SUITE("CUDA Reading Tests") {

    TEST_CASE("reading an empty file using minimal constructor") {

        CUDAReader<double> reader;

        reader.open(EMPTY_FILE, O_RDONLY | O_DIRECT);

        CHECK(reader.getFileOpened());

        CHECK(reader.getNumItemsInFile() == 0);

        CHECK(reader.getFileSize() == 0);

        double *gpuData = reader.readToGPU();

        CHECK(gpuData == nullptr);

        reader.close();

        CHECK_FALSE(reader.getFileOpened());
    }

    TEST_CASE("reading an empty file using file-based constructor") {

        CUDAReader<double> reader(EMPTY_FILE);

        CHECK(reader.getFileOpened());

        CHECK(reader.getNumItemsInFile() == 0);

        CHECK(reader.getFileSize() == 0);

        double *gpuData = reader.readToGPU();

        CHECK(gpuData == nullptr);

        reader.close();

        CHECK_FALSE(reader.getFileOpened());
    }

    TEST_CASE("reading 1 million doubles that can fit into GPU") {

        const std::size_t numItems = ONE_MILLION;

        const std::size_t numBytes = numItems * sizeof(double);

        // Read expected values using ordinary CPU I/O.
        std::vector<double> expected =
            readExpectedValues<double>(ONE_MILLION_DOUBLES_FILE, numItems);

        SUBCASE("Using file-based constructor") {

            // Read the same file directly into GPU memory using GDS.
            CUDAReader<double> reader(ONE_MILLION_DOUBLES_FILE);

            CHECK(reader.getFileOpened());

            CHECK(reader.getNumItemsInFile() == numItems);

            CHECK(reader.getFileSize() == static_cast<long>(numBytes));

            double *gpuData = reader.readToGPU();

            REQUIRE(gpuData != nullptr);

            // Verify that readToGPU() really returned device memory.
            cudaPointerAttributes attributes{};
            cudaError_t cudaStatus = cudaPointerGetAttributes(&attributes, gpuData);
            REQUIRE(cudaStatus == cudaSuccess);
            CHECK(attributes.type == cudaMemoryTypeDevice);

            // Copy GPU data back to CPU ONLY for verification.
            std::vector<double> actual = copyGPUDataToDeviceMemory(gpuData, numItems);
            checkValuesEqual(actual, expected);

            reader.close();

            CHECK_FALSE(reader.getFileOpened());
        }

        SUBCASE("Using the minimal constructor") {
            CUDAReader<double> reader;

            reader.open(ONE_MILLION_DOUBLES_FILE, O_RDONLY | O_DIRECT);

            CHECK(reader.getFileOpened());

            CHECK(reader.getNumItemsInFile() == numItems);

            CHECK(reader.getFileSize() == static_cast<long>(numBytes));

            double *gpuData = reader.readToGPU();

            REQUIRE(gpuData != nullptr);

            // Verify that readToGPU() really returned device memory.
            cudaPointerAttributes attributes{};
            cudaError_t cudaStatus = cudaPointerGetAttributes(&attributes, gpuData);
            REQUIRE(cudaStatus == cudaSuccess);
            CHECK(attributes.type == cudaMemoryTypeDevice);

            // Copy GPU data back to CPU ONLY for verification.
            std::vector<double> actual = copyGPUDataToDeviceMemory(gpuData, numItems);

            // Verify all values.
            checkValuesEqual(actual, expected);

            reader.close();

            CHECK_FALSE(reader.getFileOpened());
        }
    }

    TEST_CASE("reading 3 billion doubles that may not fit into GPU") {


    }

}
