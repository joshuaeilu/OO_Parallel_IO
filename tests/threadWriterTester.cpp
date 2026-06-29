/* readerTester.cpp tests the ParallelReader template.
 *
 * Joel C. Adams, Calvin University, Fall 2023.
 *
 * Usage
 * Build: mpic++ -fopenmp threadWriterTester.cpp -o threadWriterTester
 * Run: ./threadWriterTester
 *         change number of threads in the num_threads(P) function; 
 */

#include "DoubleWriterTester.h"
// #include "IntReaderTester.h"
#include "CharWriterTester.h"

int main() {
   const int DOUBLEFILESIZE = 6 * sizeof(double); // 6 doubles
   const int CHARFILESIZE = 6 * sizeof(char);

   #pragma omp parallel num_threads(2)
   {
        DoubleWriterTester dwt;
        ThreadWriter<double> writer("./files/6doubles.bin", omp_get_thread_num(), omp_get_num_threads(), DOUBLEFILESIZE);
        dwt.runTests(writer);

        CharWriterTester cwt;
        ThreadWriter<char> writer2("./files/6chars_output.bin", omp_get_thread_num(), omp_get_num_threads(), CHARFILESIZE);
        cwt.runTests(writer2);
   }

}

