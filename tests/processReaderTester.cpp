/* processReaderTester.cpp provides basic tests for the MPIProcessReader template.
 *
 * Joel C. Adams, Calvin University, Fall 2023.
 *
 * Usage
 * Build: mpic++ processReaderTester.cpp -o processReaderTester
 * Run: mpirun -np <P> ./processReaderTester
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
   MPIProcessReader<double> mpiDoubleReader;
   mpiDoubleReader.open("./files/5doubles.bin");
   drt.runTests(mpiDoubleReader);
   mpiDoubleReader.close();

   IntReaderTester irt;
   MPIProcessReader<int> mpiIntReader;
   mpiIntReader.open("./files/12ints.bin");
   irt.runTests(mpiIntReader);
   mpiIntReader.close();

   CharReaderTester crt;
   MPIProcessReader<char> mpiCharReader;
   mpiCharReader.open("./files/6chars.bin");
   crt.runTests(mpiCharReader);
   mpiCharReader.close();

   MPI_Finalize();

}
