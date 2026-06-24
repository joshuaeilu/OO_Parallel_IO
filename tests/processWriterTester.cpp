/* readerTester.cpp tests the ParallelReader template.
 *
 * Joel C. Adams, Calvin University, Fall 2023.
 *
 */

#include "DoubleWriterTester.h"
// #include "IntReaderTester.h"
#include "CharWriterTester.h"

int main(int argc, char** argv) {
   MPI_Init(&argc, &argv);

   DoubleWriterTester dwt;
   MPIProcessWriter<double> writer("./files/6doubles.bin");
   dwt.runTests(writer);
/*
   IntWriterTester iwt;
   iwt.runTests();
// */
   CharWriterTester cwt;
   MPIProcessWriter<char> writer2("./files/6chars.bin");
   cwt.runTests(writer2);

   MPI_Finalize();
}

