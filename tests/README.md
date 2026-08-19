# tests

This folder contains the test programs for the different I/O backends provided by `OO_Parallel_IO`.

The main test programs are:

* `mpiTests.cpp` — tests the MPI-based reader and writer implementations.
* `threadTests.cpp` — tests the thread-based reader and writer implementations.
* `cudaTests.cu` — tests the CUDA and NVIDIA GPUDirect Storage (GDS) reader and writer implementations, including:

Additional reader and writer test files may be used to test specific data types or backend functionality.

The provided `Makefile` can be used to build the test programs.

## Test Files

The `files` directory contains the text and binary files used by the tests.

Some large binary test files are not intended to be stored in the repository and may need to be generated locally. The utilities in:

`../genTextAndBinaryFiles`

can be used to create the required test files.

Binary files may also need to be regenerated when testing on a system with a different data representation or environment.
