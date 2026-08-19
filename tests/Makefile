# -------------------------------------------------------------
# OO_Parallel_IO Test Makefile
# -------------------------------------------------------------

CXX     := g++
MPICXX  := mpic++
NVCC    := nvcc


# -------------------------------------------------------------
# Compiler flags
# -------------------------------------------------------------

MPI_FLAGS := -O3 -std=c++20 -Wall -Wextra \
             -I../OO_IO/include

THREAD_FLAGS := -O3 -std=c++20 -Wall -Wextra \
                -fopenmp -pthread \
                -I../OO_IO/include

NVCC_FLAGS := -O3 -std=c++17 \
              -I../OO_IO/include \
              -I/usr/local/cuda/include

CUDA_LIBS := -L/usr/local/cuda/lib64 \
             -lcufile \
             -lcudart


# -------------------------------------------------------------
# Test executables
# -------------------------------------------------------------

MPI_TEST    := mpiTests
THREAD_TEST := threadTests
CUDA_TEST   := cudaTests

MPI_PROCS ?= 4


# -------------------------------------------------------------
# Header dependencies
# -------------------------------------------------------------

HEADERS := $(wildcard ../OO_IO/include/*.h)


# -------------------------------------------------------------
# Default target
# -------------------------------------------------------------

all: test


# -------------------------------------------------------------
# Build all tests
# -------------------------------------------------------------

build: $(MPI_TEST) $(THREAD_TEST) $(CUDA_TEST)


# -------------------------------------------------------------
# MPI tests
# -------------------------------------------------------------

$(MPI_TEST): mpiTests.cpp $(HEADERS)
	$(MPICXX) $(MPI_FLAGS) -o $@ $<


# -------------------------------------------------------------
# Threads tests
# -------------------------------------------------------------

$(THREAD_TEST): threadTests.cpp $(HEADERS)
	$(CXX) $(THREAD_FLAGS) -o $@ $<


# -------------------------------------------------------------
# CUDA tests
# -------------------------------------------------------------

$(CUDA_TEST): cudaTests.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(CUDA_LIBS)


# -------------------------------------------------------------
# Run individual test suites
# -------------------------------------------------------------

run-mpi: $(MPI_TEST)
	mpirun -np $(MPI_PROCS) ./$(MPI_TEST)


run-threads: $(THREAD_TEST)
	./$(THREAD_TEST)


run-cuda: $(CUDA_TEST)
	./$(CUDA_TEST)


# -------------------------------------------------------------
# Run all tests
# -------------------------------------------------------------

test: run-mpi run-threads run-cuda


# -------------------------------------------------------------
# Clean generated executables/files
# -------------------------------------------------------------

clean:
	rm -f $(MPI_TEST)
	rm -f $(THREAD_TEST)
	rm -f $(CUDA_TEST)
	rm -f cufile.log
	rm -f cuda-writer*.bin


.PHONY: all build test run-mpi run-threads run-cuda clean