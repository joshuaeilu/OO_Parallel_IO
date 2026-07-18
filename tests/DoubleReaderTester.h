/* DoubleReaderTester.h declares the class that tests ParallelReader
 *   using double values.
 *
 * @author: Joel C. Adams, Calvin University, Fall 2023
 */

#include <iostream>                // cout, ...
#include <fstream>                 // ifstream, ofstream, fstream
#include <mpi.h>                   // MPI types
#include <omp.h>
#include <cassert>                 // assert()
#include "../OO_IO/include/MPIProcessesIO.h"          // Reader
using namespace std;

class DoubleReaderTester {
public:
  DoubleReaderTester();
  template <typename ReaderType>
  void runTests(ReaderType & reader);
  template <typename ReaderType>
  void runFileTests(const ReaderType& reader);
  template <typename ReaderType>
  void runChunkTests(const ReaderType& reader);
  template <typename ReaderType>
  void runReadTests(ReaderType& reader);
private:
   const int MASTER = 0;
   int id;
   int numProcs;
};

DoubleReaderTester::DoubleReaderTester() {
}

template <typename ReaderType>
void DoubleReaderTester::runTests(ReaderType & reader) {
   // set id and getnumpes to support both mpi and mpi
   id = reader.getID();
   numProcs = reader.getNumPEs();
   if (id == MASTER) cout << "\nTesting Reader using doubles...\n"
                          << flush;
   runFileTests(reader);
   runReadTests(reader);
   runChunkTests(reader);

   if (id == MASTER) cout << "All double tests passed!\n" << endl;
}

template <typename ReaderType>
void DoubleReaderTester::
runFileTests(const ReaderType &reader) {
   if (id == MASTER) cout << "- Running getter tests..." << flush;
   
   assert( reader.getID() == id );
   assert( reader.getNumPEs() == numProcs );
   assert( reader.getFileName() == "./files/5doubles.bin" );
   assert( reader.getItemSize() == 8 );          // 8 bytes in a double

   
   if (id == MASTER) cout << " Passed! " << endl;
   
}

 template <typename ReaderType>
void DoubleReaderTester::
runChunkTests(const ReaderType &reader) {
   if (id == MASTER) cout << "- Running chunk-related tests... "  << flush;
   

   switch (numProcs) {
     case 1:
           assert( reader.getChunkSize() == 5 );
//           cout << " 0a " << flush;
           assert( reader.getFirstItemOffset() == 0 );
           assert( reader.getFirstByteOffset() == 0 );
//           cout << " 0b " << flush;
           break;
     case 2:
           switch (id) {
             case 0: assert( reader.getChunkSize() == 3 );
//                     cout << " 0a " << flush;
                     assert( reader.getFirstItemOffset() == 0 );
                     assert( reader.getFirstByteOffset() == 0 );
//                     cout << " 0b " << flush;
                     break;
             case 1: assert( reader.getChunkSize() == 2 );
//                     cout << " 0c " << flush;
                     assert( reader.getFirstItemOffset() == 3 );
                     assert( reader.getFirstByteOffset() == 24 );
//                     cout << " 0d " << flush;
                     break;
           }
           break;
     case 3:
           switch (id) {
             case 0: assert( reader.getChunkSize() == 2 );
//                     cout << " 0a " << flush;
                     assert( reader.getFirstItemOffset() == 0 );
                     assert( reader.getFirstByteOffset() == 0 );
//                     cout << " 0b " << flush;
                     break;
             case 1: assert( reader.getChunkSize() == 2 );
//                     cout << " 0c " << flush;
                     assert( reader.getFirstItemOffset() == 2 );
                     assert( reader.getFirstByteOffset() == 16 );
//                     cout << " 0d " << flush;
                     break;
             case 2: assert( reader.getChunkSize() == 1 );
//                     cout << " 0e " << flush;
                     assert( reader.getFirstItemOffset() == 4 );
                     assert( reader.getFirstByteOffset() == 32 );
//                     cout << " 0f " << flush;
                     break;
           }
           break;
     case 4:
           switch (id) {
             case 0: assert( reader.getChunkSize() == 2 );
//                     cout << " 0a " << flush;
                     assert( reader.getFirstItemOffset() == 0 );
                     assert( reader.getFirstByteOffset() == 0 );
//                     cout << " 0b " << flush;
                     break;
             case 1: assert( reader.getChunkSize() == 1 );
//                     cout << " 0c " << flush;
                     assert( reader.getFirstItemOffset() == 2 );
                     assert( reader.getFirstByteOffset() == 16 );
//                     cout << " 0d " << flush;
                     break;
             case 2: assert( reader.getChunkSize() == 1 );
//                     cout << " 0e " << flush;
                     assert( reader.getFirstItemOffset() == 3 );
                     assert( reader.getFirstByteOffset() == 24 );
//                     cout << " 0f " << flush;
                     break;
             case 3: assert( reader.getChunkSize() == 1 );
//                     cout << " 0g " << flush;
                     assert( reader.getFirstItemOffset() == 4 );
                     assert( reader.getFirstByteOffset() == 32 );
//                     cout << " 0h " << flush;
                     break;
           }
           break;
      case 5:
           switch (id) {
             case 0:
               assert( reader.getChunkSize() == 1 );
//               cout << " 0a " << flush;
               assert( reader.getFirstItemOffset() == 0 );
               assert( reader.getFirstByteOffset() == 0 );
//               cout << " 0b " << flush;
               break;
             case 1:
               assert( reader.getChunkSize() == 1 );
//               cout << " 0c " << flush;
               assert (reader.getFirstItemOffset() == 1 );
               assert (reader.getFirstByteOffset() == 8 );
//               cout << " 0d " << flush;
               break;
             case 2:
               assert( reader.getChunkSize() == 1 );
//               cout << " 0e " << flush;
               assert (reader.getFirstItemOffset() == 2 );
               assert (reader.getFirstByteOffset() == 16 );
//               cout << " 0f " << flush;
               break;
             case 3:
               assert( reader.getChunkSize() == 1 );
//               cout << " 0g " << flush;
               assert (reader.getFirstItemOffset() == 3 );
               assert (reader.getFirstByteOffset() == 24 );
//               cout << " 0h " << flush;
               break;
             case 4:
               assert( reader.getChunkSize() == 1 );
//               cout << " 0i " << flush;
               assert (reader.getFirstItemOffset() == 4 );
               assert (reader.getFirstByteOffset() == 32 );
//               cout << " 0j " << flush;
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

int approximatelyEqual(double v1, double v2) {
   const double THRESHOLD = 0.0000000000001;

   return abs(v2-v1) < THRESHOLD;
}

template <typename ReaderType>
void DoubleReaderTester::
runReadTests(ReaderType &reader) {
   
   if (id == MASTER) cout << "- Running read() tests... " << flush;

   auto v1 = reader.readChunk();

   assert( reader.getFileSize() == 40 );         // 8 x 5
   assert( reader.getNumItemsInFile() == 5 );
   const int SIZE = reader.getNumItemsInFile();
   vector<double> v2;
   double dVal;
   ifstream fin("files/5doubles.txt");
   assert( fin.is_open() );
   for (int i = 0; i < SIZE; ++i) {
      fin >> dVal;
      v2.push_back(dVal);
   } 
   fin.close();

   long start = -1, stop = -1;
   getChunkStartStopValues(id, numProcs, SIZE, start, stop);
// printf("\n\nv1.size(): %lu, (stop-start): %ld\n\n", v1.size(), (stop-start));
   assert( v1.size() == (stop - start) );
   int j = start;
   for (int i = 0; i < v1.size(); ++i) {
      assert( approximatelyEqual(v1[i], v2[j]) );
//        printf("Process %d: v1[%d]: %lf; v2[%d]: %lf\n",
//                      id, i, v1[i], i, v2[i]);
      ++j;
   }

   
   if (id == MASTER) cout << " Passed!" << endl;
}

