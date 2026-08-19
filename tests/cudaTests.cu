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

// const std::string ONE_MILLION_DOUBLES_FILE = "../files/1m-doubles.bin";

const std::string TWO_BILLION_DOUBLES_FILE = "../files/2b-doubles.bin";

// constexpr std::size_t ONE_MILLION = 1'000'000ULL;

// -----------------------------------------------------------------------------
// Helper: read expected values using ordinary CPU I/O
// -----------------------------------------------------------------------------

template <class ItemType>
std::vector<ItemType> readExpectedValues(const std::string &fileName,
                                         std::size_t numItems) {
    std::vector<ItemType> values(numItems);

    std::ifstream input(fileName, std::ios::binary);

    REQUIRE(input.is_open());

    const std::size_t numBytes = numItems * sizeof(ItemType);

    input.read(reinterpret_cast<char *>(values.data()),
               static_cast<std::streamsize>(numBytes));

    REQUIRE(input.gcount() == static_cast<std::streamsize>(numBytes));

    return values;
}

// -----------------------------------------------------------------------------
// Helper: copy GPU data back to CPU for test verification
// -----------------------------------------------------------------------------

template <class ItemType>
std::vector<ItemType> copyGPUDataToDeviceMemory(const ItemType *devicePtr,
                                                std::size_t numItems) {
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
void checkValuesEqual(const std::vector<ItemType> &actual,
                      const std::vector<ItemType> &expected) {
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

// // =============================================================================
// // CUDA Environment Tests
// // =============================================================================

// TEST_SUITE("CUDA Environment Tests") {

//     TEST_CASE("CUDA device is available") {

//         int deviceCount = 0;

//         cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

//         REQUIRE(cudaStatus == cudaSuccess);

//         CHECK(deviceCount > 0);
//     }
// }

// // =============================================================================
// // CUDA File Tests
// // =============================================================================

// TEST_SUITE("CUDA File Tests") {

//     TEST_CASE("minimal CUDAReader constructor") {

//         CUDAReader<double> reader;

//         CHECK(reader.getID() == 0);
//         CHECK(reader.getNumPEs() == 1);

//         CHECK(reader.getFileName() == "");

//         CHECK(reader.getItemSize() == sizeof(double));

//         CHECK_FALSE(reader.getFileOpened());
//     }

//     TEST_CASE("file-based CUDAReader constructor") {

//         CUDAReader<double> reader(FIVE_DOUBLES_FILE);

//         CHECK(reader.getID() == 0);
//         CHECK(reader.getNumPEs() == 1);

//         CHECK(reader.getFileName() == FIVE_DOUBLES_FILE);

//         CHECK(reader.getItemSize() == sizeof(double));

//         CHECK(reader.getFileSize() == 5L * static_cast<long>(sizeof(double)));

//         CHECK(reader.getNumItemsInFile() == 5);

//         CHECK(reader.getFileOpened());

//         reader.close();

//         CHECK_FALSE(reader.getFileOpened());
//     }

//     TEST_CASE("open and close reader using minimal constructor") {

//         CUDAReader<double> reader;

//         CHECK_FALSE(reader.getFileOpened());

//         reader.open(FIVE_DOUBLES_FILE, O_RDONLY | O_DIRECT);

//         CHECK(reader.getFileOpened());

//         CHECK(reader.getFileName() == FIVE_DOUBLES_FILE);

//         CHECK(reader.getItemSize() == sizeof(double));

//         CHECK(reader.getFileSize() == 5L * static_cast<long>(sizeof(double)));

//         CHECK(reader.getNumItemsInFile() == 5);

//         reader.close();

//         CHECK_FALSE(reader.getFileOpened());
//     }

//     TEST_CASE("invalid write-only access mode is rejected") {

//         CUDAReader<double> reader;

//         CHECK_THROWS_AS(reader.open(FIVE_DOUBLES_FILE, O_WRONLY | O_DIRECT),
//                         std::invalid_argument);
//     }

//     TEST_CASE("reading without opening a file throws") {

//         CUDAReader<double> reader;

//         CHECK_THROWS_AS(reader.readToGPU(), std::runtime_error);
//     }
// }

// // =============================================================================
// // CUDA Reading Tests
// // =============================================================================

// TEST_SUITE("CUDA Reading Tests") {

//     TEST_CASE("reading an empty file using minimal constructor") {

//         CUDAReader<double> reader;

//         reader.open(EMPTY_FILE, O_RDONLY | O_DIRECT);

//         CHECK(reader.getFileOpened());

//         CHECK(reader.getNumItemsInFile() == 0);

//         CHECK(reader.getFileSize() == 0);

//         double *gpuData = reader.readToGPU();

//         CHECK(gpuData == nullptr);

//         reader.close();

//         CHECK_FALSE(reader.getFileOpened());
//     }

//     TEST_CASE("reading an empty file using file-based constructor") {

//         CUDAReader<double> reader(EMPTY_FILE);

//         CHECK(reader.getFileOpened());

//         CHECK(reader.getNumItemsInFile() == 0);

//         CHECK(reader.getFileSize() == 0);

//         double *gpuData = reader.readToGPU();

//         CHECK(gpuData == nullptr);

//         reader.close();

//         CHECK_FALSE(reader.getFileOpened());
//     }

//     TEST_CASE("reading 1 million doubles that can fit into GPU") {

//         const std::size_t numItems = ONE_MILLION;

//         const std::size_t numBytes = numItems * sizeof(double);

//         // Read expected values using ordinary CPU I/O.
//         std::vector<double> expected =
//             readExpectedValues<double>(ONE_MILLION_DOUBLES_FILE, numItems);

//         SUBCASE("Using file-based constructor") {

//             // Read the same file directly into GPU memory using GDS.
//             CUDAReader<double> reader(ONE_MILLION_DOUBLES_FILE);

//             CHECK(reader.getFileOpened());

//             CHECK(reader.getNumItemsInFile() == numItems);

//             CHECK(reader.getFileSize() == static_cast<long>(numBytes));

//             double *gpuData = reader.readToGPU();

//             REQUIRE(gpuData != nullptr);

//             // Verify that readToGPU() really returned device memory.
//             cudaPointerAttributes attributes{};
//             cudaError_t cudaStatus = cudaPointerGetAttributes(&attributes, gpuData);
//             REQUIRE(cudaStatus == cudaSuccess);
//             CHECK(attributes.type == cudaMemoryTypeDevice);

//             // Copy GPU data back to CPU ONLY for verification.
//             std::vector<double> actual = copyGPUDataToDeviceMemory(gpuData, numItems);
//             checkValuesEqual(actual, expected);

//             reader.close();

//             CHECK_FALSE(reader.getFileOpened());
//         }

//         SUBCASE("Using the minimal constructor") {
//             CUDAReader<double> reader;

//             reader.open(ONE_MILLION_DOUBLES_FILE, O_RDONLY | O_DIRECT);

//             CHECK(reader.getFileOpened());

//             CHECK(reader.getNumItemsInFile() == numItems);

//             CHECK(reader.getFileSize() == static_cast<long>(numBytes));

//             double *gpuData = reader.readToGPU();

//             REQUIRE(gpuData != nullptr);

//             // Verify that readToGPU() really returned device memory.
//             cudaPointerAttributes attributes{};
//             cudaError_t cudaStatus = cudaPointerGetAttributes(&attributes, gpuData);
//             REQUIRE(cudaStatus == cudaSuccess);
//             CHECK(attributes.type == cudaMemoryTypeDevice);

//             // Copy GPU data back to CPU ONLY for verification.
//             std::vector<double> actual = copyGPUDataToDeviceMemory(gpuData, numItems);

//             // Verify all values.
//             checkValuesEqual(actual, expected);

//             reader.close();

//             CHECK_FALSE(reader.getFileOpened());
//         }
//     }

//     TEST_CASE("reading 2 billion doubles that may not fit into GPU") {

//         SUBCASE("using the normal readToGPU() to trigger out-of-memory error") {
//             CUDAReader<double> reader(TWO_BILLION_DOUBLES_FILE);

//             CHECK(reader.getFileOpened());

//             CHECK(reader.getNumItemsInFile() == 2'000'000'000);

//             CHECK(reader.getFileSize() ==
//                   static_cast<long>(2'000'000'000 * sizeof(double)));

//             // Expect an out-of-memory error when trying to read the entire file into
//             GPU
//             // memory.
//             CHECK_THROWS_AS(reader.readToGPU(), std::runtime_error);

//             reader.close();

//             CHECK_FALSE(reader.getFileOpened());
//         }

//         SUBCASE("using readChunksToGPU() to handle large file in chunks") {
//             CUDAReader<double> reader(TWO_BILLION_DOUBLES_FILE);

//             CHECK(reader.getFileOpened());

//             CHECK(reader.getNumItemsInFile() == 2'000'000'000);

//             CHECK(reader.getFileSize() ==
//                   static_cast<long>(2'000'000'000 * sizeof(double)));

//             std::vector<double> actual = reader.readChunksToGPU();

//             // Optionally, you can verify some properties of the data here.
//             // For example, check the size of the returned vector.
//             CHECK(actual.size() == 2'000'000'000);

//             reader.close();

//             CHECK_FALSE(reader.getFileOpened());
//         }

//         SUBCASE("using readChunksToGPU() with a specified Callback function") {
//             CUDAReader<double> reader(TWO_BILLION_DOUBLES_FILE);

//             CHECK(reader.getFileOpened());

//             CHECK(reader.getNumItemsInFile() == 2'000'000'000);

//             CHECK(reader.getFileSize() ==
//                   static_cast<long>(2'000'000'000 * sizeof(double)));

//             auto callback = [](double *gpuData, size_t numItems) {
//                 // Example callback: copy GPU data back to CPU and verify size.
//                 std::vector<double> actual = copyGPUDataToDeviceMemory(gpuData,
//                 numItems); CHECK(actual.size() == numItems);
//             };

//             reader.readChunksToGPU(callback);

//             reader.close();

//             CHECK_FALSE(reader.getFileOpened());
//         }

//         SUBCASE("using readChunksToGPU() with a callback to double values") {
//             CUDAReader<double> reader(FIVE_DOUBLES_FILE);

//             auto callback = [](double *gpuData, size_t numItems) {
//                 // Save the original values before modifying them.
//                 std::vector<double> original =
//                     copyGPUDataToDeviceMemory(gpuData, numItems);

//                 constexpr size_t threadsPerBlock = 256;

//                 size_t numBlocks = (numItems + threadsPerBlock - 1) / threadsPerBlock;

//                 // Double all values on the GPU.
//                 doubleValues<<<numBlocks, threadsPerBlock>>>(gpuData, numItems);

//                 checkResult(cudaGetLastError());
//                 checkResult(cudaDeviceSynchronize());

//                 // Copy the modified values back to the CPU.
//                 std::vector<double> actual = copyGPUDataToDeviceMemory(gpuData,
//                 numItems);

//                 REQUIRE(actual.size() == original.size());

//                 // Verify every value was doubled.
//                 for (size_t i = 0; i < numItems; ++i) {
//                     CHECK(actual[i] == original[i] * 2.0);
//                 }
//             };

//             reader.readChunksToGPU(callback);
//             reader.close();
//             CHECK_FALSE(reader.getFileOpened());
//         }
//     }
// }

// =============================================================================
// CUDA Writing Tests
// =============================================================================

TEST_SUITE("CUDA Writing Tests") {

    // -------------------------------------------------------------------------
    // Constructor tests
    // -------------------------------------------------------------------------

    TEST_CASE("minimal CUDAWriter constructor") {

        const long fileSize = 4096;

        CUDAWriter<double> writer(fileSize);

        CHECK(writer.getID() == 0);
        CHECK(writer.getNumPEs() == 1);

        CHECK(writer.getFileName() == "");

        CHECK(writer.getItemSize() == sizeof(double));

        CHECK(writer.getFileSize() == fileSize);

        CHECK(writer.getNumItemsInFile() ==
              static_cast<std::size_t>(fileSize) / sizeof(double));

        CHECK_FALSE(writer.getFileOpened());
    }

    TEST_CASE("CUDAWriter rejects negative file size") {

        CHECK_THROWS_AS(CUDAWriter<double>(-1), std::invalid_argument);
    }

    TEST_CASE("CUDAWriter rejects file size not divisible by ItemType") {

        CHECK_THROWS_AS(CUDAWriter<double>(4097), std::invalid_argument);
    }

    TEST_CASE("file-based CUDAWriter constructor") {

        const std::string fileName = "./files/cuda-writer-constructor.bin";
        std::remove(fileName.c_str());

        const long fileSize = 4096;

        CUDAWriter<double> writer(fileSize, fileName);

        CHECK(writer.getFileOpened());

        CHECK(writer.getFileName() == fileName);

        CHECK(writer.getFileSize() == fileSize);

        CHECK(writer.getNumItemsInFile() ==
              static_cast<std::size_t>(fileSize) / sizeof(double));

        writer.close();
        std::remove(fileName.c_str());

        CHECK_FALSE(writer.getFileOpened());
    }

    TEST_CASE("open and close CUDAWriter using minimal constructor") {

        const std::string fileName = "./files/cuda-writer-open-close.bin";

        std::remove(fileName.c_str());

        CUDAWriter<double> writer(4096);

        CHECK_FALSE(writer.getFileOpened());

        writer.open(fileName, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT);

        CHECK(writer.getFileOpened());

        CHECK(writer.getFileName() == fileName);

        CHECK(writer.getFileSize() == 4096);

        writer.close();

        CHECK_FALSE(writer.getFileOpened());

        std::remove(fileName.c_str());
    }

    // -------------------------------------------------------------------------
    // Writing without opening
    // -------------------------------------------------------------------------

    TEST_CASE("writeToFile throws when file is not open") {

        CUDAWriter<double> writer(4096);

        double *gpuData = nullptr;

        CHECK_FALSE(writer.getFileOpened());

        CHECK_THROWS_AS(writer.writeToFile(gpuData, 4096 / sizeof(double)),
                        std::runtime_error);
    }

    // -------------------------------------------------------------------------
    // nullptr
    // -------------------------------------------------------------------------

    TEST_CASE("writeToFile rejects nullptr for non-empty file") {

        const std::string fileName = "./files/cuda-writer-nullptr.bin";

        std::remove(fileName.c_str());

        constexpr std::size_t numItems = 4096 / sizeof(double);

        CUDAWriter<double> writer(4096, fileName);

        CHECK_THROWS_AS(writer.writeToFile(nullptr, numItems), std::invalid_argument);

        writer.close();

        std::remove(fileName.c_str());
    }

    TEST_CASE("writeToFile rejects zero items for non-empty file") {

        const std::string fileName = "./files/cuda-writer-zero-items.bin";

        std::remove(fileName.c_str());

        CUDAWriter<double> writer(4096, fileName);

        CHECK_THROWS_AS(writer.writeToFile(nullptr, 0), std::invalid_argument);

        writer.close();

        std::remove(fileName.c_str());
    }

    // -------------------------------------------------------------------------
    // Buffer size does not match configured file size
    // -------------------------------------------------------------------------

    TEST_CASE("writeToFile rejects too few items") {

        const std::string fileName = "./files/cuda-writer-too-few.bin";

        std::remove(fileName.c_str());

        constexpr std::size_t numItems = 4096 / sizeof(double);

        double *gpuData = nullptr;

        checkResult(cudaMalloc(reinterpret_cast<void **>(&gpuData), 4096));

        CUDAWriter<double> writer(4096, fileName);

        CHECK_THROWS_AS(writer.writeToFile(gpuData, numItems - 1), std::invalid_argument);

        writer.close();

        checkResult(cudaFree(gpuData));

        std::remove(fileName.c_str());
    }

    TEST_CASE("writeToFile rejects too many items") {

        const std::string fileName = "./files/cuda-writer-too-many.bin";

        std::remove(fileName.c_str());

        constexpr std::size_t numItems = 4096 / sizeof(double);

        double *gpuData = nullptr;

        checkResult(cudaMalloc(reinterpret_cast<void **>(&gpuData),
                               (numItems + 1) * sizeof(double)));

        CUDAWriter<double> writer(4096, fileName);

        CHECK_THROWS_AS(writer.writeToFile(gpuData, numItems + 1), std::invalid_argument);

        writer.close();

        checkResult(cudaFree(gpuData));

        std::remove(fileName.c_str());
    }

    // -------------------------------------------------------------------------
    // Empty output file
    // -------------------------------------------------------------------------

    TEST_CASE("writeToFile handles empty output file") {

        const std::string fileName = "./files/cuda-writer-empty.bin";

        std::remove(fileName.c_str());

        CUDAWriter<double> writer(0, fileName);

        CHECK(writer.getFileOpened());

        CHECK(writer.getFileSize() == 0);

        CHECK(writer.getNumItemsInFile() == 0);

        CHECK_NOTHROW(writer.writeToFile(nullptr, 0));

        writer.close();

        CHECK_FALSE(writer.getFileOpened());

        std::ifstream input(fileName, std::ios::binary | std::ios::ate);

        REQUIRE(input.is_open());

        CHECK(input.tellg() == 0);

        input.close();

        std::remove(fileName.c_str());
    }

    // -------------------------------------------------------------------------
    // Actual GPU -> file correctness
    // -------------------------------------------------------------------------

    TEST_CASE_TEMPLATE("writeToFile writes GPU data correctly", ItemType, double, int,
                       char) {

        const std::string fileName = "./files/cuda-writer-data.bin";

        std::remove(fileName.c_str());

        constexpr std::size_t fileSize = 4096;

        constexpr std::size_t numItems = fileSize / sizeof(ItemType);

        // ---------------------------------------------------------
        // Generate deterministic test values on the CPU
        // ---------------------------------------------------------

        std::vector<ItemType> expected(numItems);

        for (std::size_t i = 0; i < numItems; ++i) {

            expected[i] = static_cast<ItemType>((i * 37 + 11) % 97);
        }

        // ---------------------------------------------------------
        // Allocate GPU memory
        // ---------------------------------------------------------

        ItemType *gpuData = nullptr;

        checkResult(cudaMalloc(reinterpret_cast<void **>(&gpuData), fileSize));

        // ---------------------------------------------------------
        // Copy CPU -> GPU
        // ---------------------------------------------------------

        checkResult(
            cudaMemcpy(gpuData, expected.data(), fileSize, cudaMemcpyHostToDevice));

        // ---------------------------------------------------------
        // Write GPU -> file
        // ---------------------------------------------------------

        CUDAWriter<ItemType> writer(static_cast<long>(fileSize), fileName);

        CHECK(writer.getFileOpened());

        CHECK_NOTHROW(writer.writeToFile(gpuData, numItems));

        writer.close();

        CHECK_FALSE(writer.getFileOpened());

        // ---------------------------------------------------------
        // Read file back normally
        // ---------------------------------------------------------

        std::vector<ItemType> actual(numItems);

        std::ifstream input(fileName, std::ios::binary);

        REQUIRE(input.is_open());

        input.read(reinterpret_cast<char *>(actual.data()),
                   static_cast<std::streamsize>(fileSize));

        REQUIRE(input.gcount() == static_cast<std::streamsize>(fileSize));

        input.close();

        // ---------------------------------------------------------
        // Verify every item
        // ---------------------------------------------------------

        REQUIRE(actual.size() == expected.size());

        for (std::size_t i = 0; i < numItems; ++i) {

            CAPTURE(i);
            CAPTURE(actual[i]);
            CAPTURE(expected[i]);

            CHECK(actual[i] == expected[i]);
        }

        // ---------------------------------------------------------
        // Cleanup
        // ---------------------------------------------------------

        checkResult(cudaFree(gpuData));

        std::remove(fileName.c_str());
    }

    // -------------------------------------------------------------------------
    // CUDA managed memory
    // -------------------------------------------------------------------------

    TEST_CASE("writeToFile writes GPU memory correctly") {

        const std::string fileName = "./files/cuda-writer-test.bin";

        std::remove(fileName.c_str());

        constexpr std::size_t numItems = 512;
        constexpr std::size_t fileSize = numItems * sizeof(double);

        // -----------------------------------------
        // Create known CPU data
        // -----------------------------------------

        std::vector<double> expected(numItems);

        for (std::size_t i = 0; i < numItems; ++i) {
            expected[i] = static_cast<double>((i * 17 + 5) % 101);
        }

        // -----------------------------------------
        // Allocate actual GPU device memory
        // -----------------------------------------

        double *gpuData = nullptr;

        checkResult(cudaMalloc(reinterpret_cast<void **>(&gpuData), fileSize));

        // -----------------------------------------
        // Copy data to GPU
        // -----------------------------------------

        checkResult(
            cudaMemcpy(gpuData, expected.data(), fileSize, cudaMemcpyHostToDevice));

        // -----------------------------------------
        // Write GPU -> file
        // -----------------------------------------

        CUDAWriter<double> writer(static_cast<long>(fileSize), fileName);

        REQUIRE(writer.getFileOpened());

        CHECK_NOTHROW(writer.writeToFile(gpuData, numItems));

        writer.close();

        // -----------------------------------------
        // Read file back using CPU
        // -----------------------------------------

        std::vector<double> actual(numItems);

        std::ifstream input(fileName, std::ios::binary);

        REQUIRE(input.is_open());

        input.read(reinterpret_cast<char *>(actual.data()),
                   static_cast<std::streamsize>(fileSize));

        REQUIRE(input.gcount() == static_cast<std::streamsize>(fileSize));

        input.close();

        // -----------------------------------------
        // Verify contents
        // -----------------------------------------

        REQUIRE(actual.size() == expected.size());

        for (std::size_t i = 0; i < numItems; ++i) {
            CAPTURE(i);
            CHECK(actual[i] == expected[i]);
        }

        checkResult(cudaFree(gpuData));

        std::remove(fileName.c_str());
    }

    // -------------------------------------------------------------------------
    // Writing after close
    // -------------------------------------------------------------------------

    // TEST_CASE("writeToFile throws after writer is closed") {

    //     const std::string fileName = "./files/cuda-writer-after-close.bin";

    //     std::remove(fileName.c_str());

    //     constexpr std::size_t fileSize = 4096;

    //     constexpr std::size_t numItems = fileSize / sizeof(double);

    //     double *gpuData = nullptr;

    //     checkResult(cudaMalloc(reinterpret_cast<void **>(&gpuData), fileSize));

    //     CUDAWriter<double> writer(static_cast<long>(fileSize), fileName);

    //     writer.close();

    //     CHECK_FALSE(writer.getFileOpened());

    //     CHECK_THROWS_AS(writer.writeToFile(gpuData, numItems), std::runtime_error);

    //     checkResult(cudaFree(gpuData));

    //     std::remove(fileName.c_str());
    // }
}