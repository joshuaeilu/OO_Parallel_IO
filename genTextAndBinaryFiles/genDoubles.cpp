/*
 * genDoubles.cpp
 *
 * Generates N random binary double values in the range [0.0, 1.0)
 * and writes them to a binary file.
 *
 * Usage:
 *     ./generate <N> <fileName>
 *
 * Example:
 *     ./generate 3000000000 3b-doubles
 *
 * This creates:
 *     3b-doubles.bin
 *
 * 3,000,000,000 doubles * 8 bytes = 24,000,000,000 bytes
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using Item = double;

int main(int argc, char* argv[])
{
    const Item LOW  = 0.0;
    const Item HIGH = 1.0;

    // Generate and write 1 million doubles at a time.
    // This uses approximately 8 MB of RAM.
    constexpr std::uint64_t CHUNK_SIZE = 1'000'000;

    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <N> <fileName>\n";

        return EXIT_FAILURE;
    }

    // Read N as a 64-bit unsigned integer.
    std::uint64_t numItems;

    try
    {
        numItems = std::stoull(argv[1]);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: invalid number of items: "
                  << argv[1] << '\n';

        return EXIT_FAILURE;
    }

    if (numItems == 0)
    {
        std::cerr << "Error: N must be greater than 0.\n";
        return EXIT_FAILURE;
    }

    const std::string binFileName =
        std::string(argv[2]) + ".bin";

    std::ofstream outFile(
        binFileName,
        std::ios::out |
        std::ios::binary |
        std::ios::trunc
    );

    if (!outFile)
    {
        std::cerr << "Error: could not open '"
                  << binFileName
                  << "' for writing.\n";

        return EXIT_FAILURE;
    }

    // Random-number generator.
    std::random_device rd;
    std::mt19937_64 generator(rd());

    std::uniform_real_distribution<Item>
        distribution(LOW, HIGH);

    // Allocate only one chunk in memory.
    std::vector<Item> buffer(CHUNK_SIZE);

    std::uint64_t itemsWritten = 0;

    while (itemsWritten < numItems)
    {
        // The final chunk may contain fewer than CHUNK_SIZE items.
        const std::uint64_t itemsRemaining =
            numItems - itemsWritten;

        const std::uint64_t currentChunkSize =
            std::min(CHUNK_SIZE, itemsRemaining);

        // Generate the current chunk.
        for (std::uint64_t i = 0;
             i < currentChunkSize;
             ++i)
        {
            buffer[i] = distribution(generator);
        }

        // Number of bytes in this chunk.
        const std::streamsize bytesToWrite =
            static_cast<std::streamsize>(
                currentChunkSize * sizeof(Item)
            );

        // Write the chunk to the binary file.
        outFile.write(
            reinterpret_cast<const char*>(buffer.data()),
            bytesToWrite
        );

        if (!outFile)
        {
            std::cerr
                << "\nError: failed while writing to '"
                << binFileName << "'.\n";

            return EXIT_FAILURE;
        }

        itemsWritten += currentChunkSize;

        // Print progress every 100 million doubles
        // and when finished.
        if (itemsWritten % 100'000'000 == 0 ||
            itemsWritten == numItems)
        {
            const double percent =
                100.0 *
                static_cast<double>(itemsWritten) /
                static_cast<double>(numItems);

            std::cout
                << "\rGenerated "
                << itemsWritten
                << " / "
                << numItems
                << " doubles ("
                << std::fixed
                << std::setprecision(1)
                << percent
                << "%)"
                << std::flush;
        }
    }

    outFile.close();

    if (!outFile)
    {
        std::cerr
            << "\nError: failed to close file correctly.\n";

        return EXIT_FAILURE;
    }

    const std::uint64_t expectedBytes =
        numItems * sizeof(Item);

    std::cout
        << "\n\nSuccessfully generated "
        << numItems
        << " doubles.\n"
        << "Output file: "
        << binFileName
        << '\n'
        << "Expected file size: "
        << expectedBytes
        << " bytes\n";

    return EXIT_SUCCESS;
}