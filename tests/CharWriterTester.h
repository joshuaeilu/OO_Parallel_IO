/* CharWriterTester.h declares the class that tests ParallelWriter
 *   using char values.
 *
 * @author: Joel C. Adams, Calvin University, Fall 2023
 */

#include <iostream>                // cout, ...
#include <fstream>                 // ifstream, ofstream, fstream
#include <mpi.h>                   // MPI types
#include "../OO_IO/include/MPIProcessesIO.h"          // ParallelWriter
using namespace std;

class CharWriterTester {
public:
  CharWriterTester();

  template <typename WriterType>
  void runTests(WriterType& writer);
  template <typename WriterType>
  void runFileTests(const WriterType& writer);
  template <typename WriterType>
  void runWriteTests(WriterType& writer);
  template <typename WriterType>
  void runChunkTests(const WriterType& writer);
private:
   const int MASTER = 0;
   int id;
   int numProcs;
};

CharWriterTester::CharWriterTester() {

}

template <typename WriterType>
void CharWriterTester::runTests(WriterType& writer) {
  id = writer.getID();
   numProcs = writer.getNumPEs();
   if (id == MASTER) cout << "\nTesting Writer using chars...\n"
                          << flush;
   
   runFileTests(writer);
   runWriteTests(writer);
   runChunkTests(writer);

   if (id == MASTER) cout << "All char tests passed!\n" << endl;
}

template <typename WriterType>
void CharWriterTester::runFileTests(const WriterType& writer) {
   if (id == MASTER) cout << "- Running getter tests..." << flush;

   assert( writer.getID() == id );
   assert( writer.getNumPEs() == numProcs );
   assert( writer.getFileName() == "./files/6chars_output.bin" );
   assert( writer.getItemSize() == 1 );          // 1 byte in a char

   if (id == MASTER) cout << " Passed! " << endl;
}

template <typename WriterType>
void CharWriterTester::runWriteTests(WriterType& writer) {
   if (id == MASTER) cout << "- Running write() tests... " << flush;

   const int SIZE = 6;
   long start = -1, stop = -1;
   getChunkStartStopValues(id, numProcs, SIZE, start, stop);
   int chunkSize = stop - start;
   vector<char> v1(chunkSize);
   char cVal = 'a';
   for (int i = 0; i < chunkSize; ++i) {
      v1[i] = cVal;
      cVal += 2;
   }
   writer.writeChunk(v1);
   
   assert( writer.getFileSize() == 6 );         // 1 x 6
   assert( writer.getNumItemsInFile() == 6 );

   #pragma omp barrier

  ThreadReader<char> tReader(
    "./files/6chars_output.bin",
    id,
    numProcs);
   std::span<const char> v2 = tReader.readChunk();


   assert( v2.size() == v1.size() );
   for (int i = 0; i < v2.size(); ++i) {
      assert( v2[i] == v1[i] );
   }

   if (id == MASTER) cout << " Passed!" << endl;
}

template <typename WriterType>
void CharWriterTester::runChunkTests(const WriterType& writer) {
   if (id == MASTER) cout << "- Running chunk-related tests... "  << flush;

   switch (numProcs) {
     case 1:   // 6
           assert( writer.getChunkSize() == 6 );
           assert( writer.getFirstItemOffset() == 0 );
           assert( writer.getFirstByteOffset() == 0 );
           break;
     case 2:   // 3,3
           switch (id) {
             case 0: assert( writer.getChunkSize() == 3 );
                     assert( writer.getFirstItemOffset() == 0 );
                     assert( writer.getFirstByteOffset() == 0 );
                     break;
             case 1: assert( writer.getChunkSize() == 3 );
                     assert( writer.getFirstItemOffset() == 3 );
                     assert( writer.getFirstByteOffset() == 3 );
                     break;
           }
           break;
     case 3:   // 2,2,2
           switch (id) {
             case 0: assert( writer.getChunkSize() == 2 );
                     assert( writer.getFirstItemOffset() == 0 );
                     assert( writer.getFirstByteOffset() == 0 );
                     break;
             case 1: assert( writer.getChunkSize() == 2 );
                     assert( writer.getFirstItemOffset() == 2 );
                     assert( writer.getFirstByteOffset() == 2 );
                     break;
             case 2: assert( writer.getChunkSize() == 2 );
                     assert( writer.getFirstItemOffset() == 4 );
                     assert( writer.getFirstByteOffset() == 4 );
                     break;
           }
           break;
     case 4:   // 2,2,1,1
           switch (id) {
             case 0: assert( writer.getChunkSize() == 2 );
                     assert( writer.getFirstItemOffset() == 0 );
                     assert( writer.getFirstByteOffset() == 0 );
                     break;
             case 1: assert( writer.getChunkSize() == 2 );
                     assert( writer.getFirstItemOffset() == 2 );
                     assert( writer.getFirstByteOffset() == 2 );
                     break;
             case 2: assert( writer.getChunkSize() == 1 );
                     assert( writer.getFirstItemOffset() == 4 );
                     assert( writer.getFirstByteOffset() == 4 );
                     break;
             case 3: assert( writer.getChunkSize() == 1 );
                     assert( writer.getFirstItemOffset() == 5 );
                     assert( writer.getFirstByteOffset() == 5 );
                     break;
           }
           break;
      case 5:   // 2,1,1,1,1
           switch (id) {
             case 0:
               assert( writer.getChunkSize() == 2 );
               assert( writer.getFirstItemOffset() == 0 );
               assert( writer.getFirstByteOffset() == 0 );
               break;
             case 1:
               assert( writer.getChunkSize() == 1 );
               assert (writer.getFirstItemOffset() == 2 );
               assert (writer.getFirstByteOffset() == 2 );
               break;
             case 2:
               assert( writer.getChunkSize() == 1 );
               assert (writer.getFirstItemOffset() == 3 );
               assert (writer.getFirstByteOffset() == 3 );
               break;
             case 3:
               assert( writer.getChunkSize() == 1 );
               assert (writer.getFirstItemOffset() == 4 );
               assert (writer.getFirstByteOffset() == 4 );
               break;
             case 4:
               assert( writer.getChunkSize() == 1 );
               assert (writer.getFirstItemOffset() == 5 );
               assert (writer.getFirstByteOffset() == 5 );
               break;
             default:
               cerr << "\n*** Switch 2: flow should not reach here\n\n";
               MPI_Abort(MPI_COMM_WORLD, 2);
           }
           break;
      case 6:   // 1,1,1,1,1,1
           switch (id) {
             case 0:
               assert( writer.getChunkSize() == 1 );
               assert( writer.getFirstItemOffset() == 0 );
               assert( writer.getFirstByteOffset() == 0 );
               break;
             case 1:
               assert( writer.getChunkSize() == 1 );
               assert (writer.getFirstItemOffset() == 1 );
               assert (writer.getFirstByteOffset() == 1 );
               break;
             case 2:
               assert( writer.getChunkSize() == 1 );
               assert (writer.getFirstItemOffset() == 2 );
               assert (writer.getFirstByteOffset() == 2 );
               break;
             case 3:
               assert( writer.getChunkSize() == 1 );
               assert (writer.getFirstItemOffset() == 3 );
               assert (writer.getFirstByteOffset() == 3 );
               break;
             case 4:
               assert( writer.getChunkSize() == 1 );
               assert (writer.getFirstItemOffset() == 4 );
               assert (writer.getFirstByteOffset() == 4 );
               break;
             case 5:
               assert( writer.getChunkSize() == 1 );
               assert (writer.getFirstItemOffset() == 5 );
               assert (writer.getFirstByteOffset() == 5 );
               break;
             default:
               cerr << "\n*** Switch 2: flow should not reach here\n\n";
               MPI_Abort(MPI_COMM_WORLD, 2);
           }
           break;
      default:
           cerr << "\n*** Switch: flow should never reach here\n\n";
           MPI_Abort(MPI_COMM_WORLD, 1);
   }

   if (id == MASTER) cout << " Passed! " << endl;

}



