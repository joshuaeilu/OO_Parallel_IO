/* readerTester.cpp tests the ParallelReader template.
 *
 * Joel C. Adams, Calvin University, Fall 2023.\
 * 
 * Usage
 * Build: mpic++ processWriterTester.cpp -o processWriterTester
 * Run: mpirun -np <P> ./processWriterTester
 *         where P = 1, 2, or 3.
 *
 */

#include "DoubleWriterTester.h"
// #include "IntReaderTester.h"
#include "CharWriterTester.h"

int main(int argc, char** argv) {
   MPI_Init(&argc, &argv);

   DoubleWriterTester dwt;
   MPIProcessWriter<double> writer;
   writer.open("./files/6doubles.bin");
   dwt.runTests(writer);
   writer.close();
/*
   IntWriterTester iwt;
   iwt.runTests();
// */
   CharWriterTester cwt;
   MPIProcessWriter<char> writer2;
   writer2.open("./files/6chars_output.bin");
   cwt.runTests(writer2);
   writer2.close();

   MPI_Finalize();
}

