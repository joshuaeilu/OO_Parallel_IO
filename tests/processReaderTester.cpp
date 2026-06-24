/* readerTester.cpp provides basic tests for the ParallelReader template.
 *
 * Joel C. Adams, Calvin University, Fall 2023.
 *
 * Usage: mpirun -np <P> ./readerTester
 *         where P = 1, 2, or 3.
 */

/* readerTester.cpp provides basic tests for the ParallelReader template.
 *
 * Joel C. Adams, Calvin University, Fall 2023.
 *
 * Usage: mpirun -np <P> ./readerTester
 *         where P = 1, 2, or 3.
 */

#include "DoubleReaderTester.h"
#include "IntReaderTester.h"
#include "CharReaderTester.h"

int main(int argc, char **argv)
{
   // Process Tests
   MPI_Init(&argc, &argv);

   DoubleReaderTester drt;
   MPIProcessReader<double> mpiDoubleReader("./files/5doubles.bin");
   drt.runTests(mpiDoubleReader);

   IntReaderTester irt;
   MPIProcessReader<int> mpiIntReader("./files/12ints.bin");
   irt.runTests(mpiIntReader);

   CharReaderTester crt;
   MPIProcessReader<char> mpiCharReader("./files/6chars.bin");
   crt.runTests(mpiCharReader);

   MPI_Finalize();

}
