/* readerTester.cpp tests the ParallelReader template.
 *
 * Joel C. Adams, Calvin University, Fall 2023.
 *
 */

#include "DoubleWriterTester.h"
// #include "IntReaderTester.h"
#include "CharWriterTester.h"

int main() {

   #pragma omp parallel num_threads(2)
   {
        DoubleWriterTester dwt;
        ThreadWriter<double> writer("./files/6doubles.bin", omp_get_thread_num(), omp_get_num_threads());
        dwt.runTests(writer);

        CharWriterTester cwt;
        ThreadWriter<char> writer2("./files/6chars.bin", omp_get_thread_num(), omp_get_num_threads());
        cwt.runTests(writer2);
   }

}

