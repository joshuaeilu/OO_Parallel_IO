# OO_Parallel_IO

**OO_Parallel_IO** is a C++ library that provides object-oriented abstractions for parallel file I/O.

The project was inspired by Professor Joel Adams's [OO_MPI_IO](https://github.com/joeladams/OO_MPI_IO) and extends the idea to multiple parallel programming models.

## Supported Backends

OO_Parallel_IO currently provides support for:

* **MPI** — parallel I/O using multiple processes and MPI-IO
* **Threads** — parallel I/O using multiple CPU threads and `mmap()`
* **CUDA / GDS** — GPU-oriented I/O using CUDA and NVIDIA GPUDirect Storage

The library divides a file into approximately equal chunks so that multiple processing elements can access and process different parts of the file in parallel.

```text
                    File
    ┌────────┬────────┬────────┬────────┐
    │ Chunk0 │ Chunk1 │ Chunk2 │ Chunk3 │
    └───┬────┴───┬────┴───┬────┴───┬────┘
        │        │        │        │
        ▼        ▼        ▼        ▼
       PE 0     PE 1     PE 2     PE 3
```

Depending on the backend, a processing element may be an MPI process, CPU thread, or GPU operation.

## Main Classes

```cpp
MPIProcessReader<ItemType>
MPIProcessWriter<ItemType>

ThreadReader<ItemType>
ThreadWriter<ItemType>

CUDAReader<ItemType>
CUDAWriter<ItemType>
```

Common functionality is provided through the `IO_Base<ItemType>` abstraction.

## Example

Using the threads backend:

```cpp
#include "OO_IO/include/ThreadsIO.h"
#include <omp.h>

#pragma omp parallel num_threads(4)
{
    int id = omp_get_thread_num();
    int numPEs = omp_get_num_threads();

    ThreadReader<double> reader(id, numPEs);

    reader.open("data.bin");

    std::vector<double> chunk = reader.readChunk();

    // Process this thread's chunk.

    reader.close();
}
```

## Project Structure

```text
OO_Parallel_IO/
├── OO_IO/
│   └── include/
│       ├── IO_Base.h
│       ├── OO_MPI_IO.h
│       ├── ThreadsIO.h
│       └── CUDAIO.h
├── tests/
└── genTextAndBinaryFiles/
```

## Goal

The goal of OO_Parallel_IO is to make parallel I/O easier to use by hiding lower-level implementation details behind simple reader and writer abstractions.

## Author

**Joshua Eilu**
Calvin University
Summer 2026

Special thanks to **Professor Joel C. Adams** for his mentorship and for the original [OO_MPI_IO](https://github.com/joeladams/OO_MPI_IO) project that inspired this work.
