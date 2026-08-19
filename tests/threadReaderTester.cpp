/* threadReaderTester.cpp provides basic tests for the ThreadReader template.
 *
 * Joel C. Adams, Calvin University, Fall 2023.
 *
 * Usage
 * Build: mpic++ -fopenmp threadReaderTester.cpp -o threadReaderTester
 * Run: ./threadReaderTester
 *         change number of threads in the main function where INT NUM_THREADS = 5; 
 */
#include "DoubleReaderTester.h"
#include "IntReaderTester.h"
#include "CharReaderTester.h"
#include <threads.h>
#include <thread>
#include <iostream>

// ======================================================
// PTHREAD TEST SUPPORT
// ======================================================

struct ThreadArgs
{
    int id;
    int numThreads;
};

void* pthreadTests(void* arg)
{
    ThreadArgs* args = static_cast<ThreadArgs*>(arg);

    DoubleReaderTester drt;
    ThreadReader<double> doubleReader(args->id, args->numThreads);
    doubleReader.open("./files/5doubles.bin");
    drt.runTests(doubleReader);
    doubleReader.close();


    IntReaderTester irt;
    ThreadReader<int> intReader(args->id, args->numThreads);
    intReader.open("./files/12ints.bin");
    irt.runTests(intReader);
    intReader.close();

    CharReaderTester crt;
    ThreadReader<char> charReader(args->id, args->numThreads);
    charReader.open("./files/6chars.bin");
    crt.runTests(charReader);
    charReader.close();

    return nullptr;
}

// ======================================================
// CPP TEST SUPPORT
// ======================================================


void cppTests(int id, int numThreads)
{
    DoubleReaderTester drt;
    ThreadReader<double> doubleReader(id, numThreads);
    doubleReader.open("./files/5doubles.bin");
    drt.runTests(doubleReader);
    doubleReader.close();



    IntReaderTester irt;
    ThreadReader<int> intReader(id, numThreads);
    intReader.open("./files/12ints.bin");
    irt.runTests(intReader);
    intReader.close();

    CharReaderTester crt;
    ThreadReader<char> charReader(id, numThreads);
    charReader.open("./files/6chars.bin");
    crt.runTests(charReader);
    charReader.close();
}

// // ======================================================
// // MAIN
// // ======================================================

int main()
{
    const int NUM_THREADS = 5;  

    std::cout << "\n====================================\n";
    std::cout << "OPENMP TESTS\n";
    std::cout << "====================================\n";

#pragma omp parallel num_threads(NUM_THREADS)
    {
        DoubleReaderTester drt;
        ThreadReader<double> doubleReader( omp_get_thread_num(), omp_get_num_threads());
        doubleReader.open("./files/5doubles.bin");
        drt.runTests(doubleReader);
        doubleReader.close();

        IntReaderTester irt;
        ThreadReader<int> intReader(omp_get_thread_num(), omp_get_num_threads());
        intReader.open("./files/12ints.bin");
        irt.runTests(intReader);
        intReader.close();

        CharReaderTester crt;
        ThreadReader<char> charReader(omp_get_thread_num(), omp_get_num_threads());
        charReader.open("./files/6chars.bin");
        crt.runTests(charReader);
        charReader.close();

    }

    std::cout << "\nAll OPEN MP tests completed.\n";

    std::cout << "\n====================================\n";
    std::cout << "PTHREAD TESTS\n";
    std::cout << "====================================\n";

    pthread_t threads[NUM_THREADS];
    ThreadArgs args[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++)
    {
        args[i].id = i;
        args[i].numThreads = NUM_THREADS;

        pthread_create(&threads[i], nullptr, pthreadTests, &args[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], nullptr);
    }

    std::cout << "\nAll PTHREAD tests completed.\n";

    std::cout << "\n====================================\n";
    std::cout << "CPP TESTS\n";
    std::cout << "====================================\n";

    std::vector<std::thread> cpp_threads;
    cpp_threads.reserve(NUM_THREADS);
    for (int threadID = 0; threadID < NUM_THREADS; ++threadID)
    {
        cpp_threads.emplace_back(cppTests, threadID, NUM_THREADS);
    }

    //--------------------------------------------------
    // Wait for workers
    //--------------------------------------------------
    for (auto& thread : cpp_threads)
    {
        thread.join();
    }

     std::cout << "\nAll CPP tests completed.\n";

    return 0;
}