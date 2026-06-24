/* IntReaderTester.h declares the class that tests Reader
 *   using int values.
 *
 * @author: Joel C. Adams, Calvin University, Fall 2023
 */

#include <iostream>                // cout, ...
#include <fstream>                 // ifstream, ofstream, fstream
#include <mpi.h>                   // MPI types
#include "../OO_IO.h"          // 
using namespace std;

class IntReaderTester {
public:
  IntReaderTester();
  template <typename ReaderType>
  void runTests(ReaderType &reader);
  template <typename ReaderType>
  void runGetterTests(const ReaderType& reader);
  template <typename ReaderType>
  void runChunkTests(const ReaderType& reader);
  template <typename ReaderType>
  void runReadTests(ReaderType& reader);
  private:
  const int MASTER = 0;
  int id;
  int numProcs;
};

IntReaderTester::IntReaderTester() {

}


template <typename ReaderType>
void IntReaderTester::runTests(ReaderType &reader) {
     // set id and getnumpes to support both mpi and mpi
   id = reader.getID();
   numProcs = reader.getNumPEs();
   if (id == MASTER) cout << "\nTesting Reader using ints...\n"
                          << flush;
   runGetterTests(reader);
   runReadTests(reader);
   runChunkTests(reader);

   if (id == MASTER) cout << "All int tests passed!\n" << endl;
}

template <typename ReaderType>
void IntReaderTester::
runGetterTests(const ReaderType &reader) {
   if (id == MASTER) cout << "- Running getter tests..." << flush;
   

   assert( reader.getID() == id );
   assert( reader.getNumPEs() == numProcs );
   assert( reader.getFileName() == "./files/12ints.bin" );
   assert( reader.getItemSize() == 4 );          // 4 bytes in an int

   
   if (id == MASTER) cout << " Passed! " << endl;
   
}

template <typename ReaderType>
void IntReaderTester::
runChunkTests(const ReaderType &reader) {
   if (id == MASTER) cout << "- Running chunk-related tests... "  << flush;
   

   switch (numProcs) {
     case 1:
           assert( reader.getChunkSize() == 12 );
//           cout << " 0a " << flush;
           assert( reader.getFirstItemOffset() == 0 );
           assert( reader.getFirstByteOffset() == 0 );
//           cout << " 0b " << flush;
           break;
     case 2:
           switch (id) {
             case 0: assert( reader.getChunkSize() == 6 );
//                     cout << " 0a " << flush;
                     assert( reader.getFirstItemOffset() == 0 );
                     assert( reader.getFirstByteOffset() == 0 );
//                     cout << " 0b " << flush;
                     break;
             case 1: assert( reader.getChunkSize() == 6 );
//                     cout << " 0c " << flush;
                     assert( reader.getFirstItemOffset() == 6 );
                     assert( reader.getFirstByteOffset() == 24 );
//                     cout << " 0d " << flush;
                     break;
           }
           break;
     case 3:
           switch (id) {
             case 0: assert( reader.getChunkSize() == 4 );
//                     cout << " 0a " << flush;
                     assert( reader.getFirstItemOffset() == 0 );
                     assert( reader.getFirstByteOffset() == 0 );
//                     cout << " 0b " << flush;
                     break;
             case 1: assert( reader.getChunkSize() == 4 );
//                     cout << " 0c " << flush;
                     assert( reader.getFirstItemOffset() == 4 );
                     assert( reader.getFirstByteOffset() == 16 );
//                     cout << " 0d " << flush;
                     break;
             case 2: assert( reader.getChunkSize() == 4 );
//                     cout << " 0e " << flush;
                     assert( reader.getFirstItemOffset() == 8 );
                     assert( reader.getFirstByteOffset() == 32 );
//                     cout << " 0f " << flush;
                     break;
           }
           break;
     case 4:
           switch (id) {
             case 0: assert( reader.getChunkSize() == 3 );
//                     cout << " 0a " << flush;
                     assert( reader.getFirstItemOffset() == 0 );
                     assert( reader.getFirstByteOffset() == 0 );
//                     cout << " 0b " << flush;
                     break;
             case 1: assert( reader.getChunkSize() == 3 );
//                     cout << " 0c " << flush;
                     assert( reader.getFirstItemOffset() == 3 );
                     assert( reader.getFirstByteOffset() == 12 );
//                     cout << " 0d " << flush;
                     break;
             case 2: assert( reader.getChunkSize() == 3 );
//                     cout << " 0e " << flush;
                     assert( reader.getFirstItemOffset() == 6 );
                     assert( reader.getFirstByteOffset() == 24 );
//                     cout << " 0f " << flush;
                     break;
             case 3: assert( reader.getChunkSize() == 3 );
//                     cout << " 0g " << flush;
                     assert( reader.getFirstItemOffset() == 9 );
                     assert( reader.getFirstByteOffset() == 36 );
//                     cout << " 0h " << flush;
                     break;
           }
           break;
      case 5:
           switch (id) {
             case 0:
               assert( reader.getChunkSize() == 3 );
//               cout << " 0a " << flush;
               assert( reader.getFirstItemOffset() == 0 );
               assert( reader.getFirstByteOffset() == 0 );
//               cout << " 0b " << flush;
               break;
             case 1:
               assert( reader.getChunkSize() == 3 );
//               cout << " 0c " << flush;
               assert (reader.getFirstItemOffset() == 3 );
               assert (reader.getFirstByteOffset() == 12 );
//               cout << " 0d " << flush;
               break;
             case 2:
               assert( reader.getChunkSize() == 2 );
//               cout << " 0e " << flush;
               assert (reader.getFirstItemOffset() == 6 );
               assert (reader.getFirstByteOffset() == 24 );
//               cout << " 0f " << flush;
               break;
             case 3:
               assert( reader.getChunkSize() == 2 );
//               cout << " 0g " << flush;
               assert (reader.getFirstItemOffset() == 8 );
               assert (reader.getFirstByteOffset() == 32 );
//               cout << " 0h " << flush;
               break;
             case 4:
               assert( reader.getChunkSize() == 2 );
//               cout << " 0i " << flush;
               assert (reader.getFirstItemOffset() == 10 );
               assert (reader.getFirstByteOffset() == 40 );
//               cout << " 0j " << flush;
               break;
             default:
               cerr << "\n*** Switch 2: flow should not reach here\n\n";
               MPI_Abort(MPI_COMM_WORLD, 2);
          }
          break;
      case 6:
          switch (id) {
             case 0:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 0 );
               assert (reader.getFirstByteOffset() == 0 );
               break;
             case 1:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 2 );
               assert (reader.getFirstByteOffset() == 8 );
               break;
             case 2:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 4 );
               assert (reader.getFirstByteOffset() == 16 );
               break;
             case 3:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 6 );
               assert (reader.getFirstByteOffset() == 24 );
               break;
             case 4:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 8 );
               assert (reader.getFirstByteOffset() == 32 );
               break;
             case 5:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 10 );
               assert (reader.getFirstByteOffset() == 40 );
               break;
          }
          break;
      case 7:            // 2,2,2,2,2,1,1
          switch (id) {
             case 0:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 0 );
               assert (reader.getFirstByteOffset() == 0 );
               break;
             case 1:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 2 );
               assert (reader.getFirstByteOffset() == 8 );
               break;
             case 2:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 4 );
               assert (reader.getFirstByteOffset() == 16 );
               break;
             case 3:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 6 );
               assert (reader.getFirstByteOffset() == 24 );
               break;
             case 4:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 8 );
               assert (reader.getFirstByteOffset() == 32 );
               break;
             case 5:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 10 );
               assert (reader.getFirstByteOffset() == 40 );
               break;
             case 6:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 11 );
               assert (reader.getFirstByteOffset() == 44 );
               break;
          }
          break;
      case 8:        // 2, 2, 2, 2, 1, 1, 1, 1
          switch (id) {
             case 0:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 0 );
               assert (reader.getFirstByteOffset() == 0 );
               break;
             case 1:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 2 );
               assert (reader.getFirstByteOffset() == 8 );
               break;
             case 2:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 4 );
               assert (reader.getFirstByteOffset() == 16 );
               break;
             case 3:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 6 );
               assert (reader.getFirstByteOffset() == 24 );
               break;
             case 4:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 8 );
               assert (reader.getFirstByteOffset() == 32 );
               break;
             case 5:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 9 );
               assert (reader.getFirstByteOffset() == 36);
               break;
             case 6:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 10 );
               assert (reader.getFirstByteOffset() == 40 );
               break;
             case 7:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 11 );
               assert (reader.getFirstByteOffset() == 44 );
          }
          break;
      case 9:        // 2, 2, 2, 1, 1, 1, 1, 1, 1
          switch (id) {
             case 0:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 0 );
               assert (reader.getFirstByteOffset() == 0 );
               break;
             case 1:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 2 );
               assert (reader.getFirstByteOffset() == 8 );
               break;
             case 2:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 4 );
               assert (reader.getFirstByteOffset() == 16 );
               break;
             case 3:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 6 );
               assert (reader.getFirstByteOffset() == 24 );
               break;
             case 4:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 7 );
               assert (reader.getFirstByteOffset() == 28 );
               break;
             case 5:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 8 );
               assert (reader.getFirstByteOffset() == 32 );
               break;
             case 6:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 9 );
               assert (reader.getFirstByteOffset() == 36 );
               break;
             case 7:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 10 );
               assert (reader.getFirstByteOffset() == 40 );
               break;
             case 8:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 11 );
               assert (reader.getFirstByteOffset() == 44 );
               break;
          }
          break;
      case 10:        // 2, 2, 1, 1, 1, 1, 1, 1, 1, 1
          switch (id) {
             case 0:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 0 );
               assert (reader.getFirstByteOffset() == 0 );
               break;
             case 1:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 2 );
               assert (reader.getFirstByteOffset() == 8 );
               break;
             case 2:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 4 );
               assert (reader.getFirstByteOffset() == 16 );
               break;
             case 3:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 5 );
               assert (reader.getFirstByteOffset() == 20 );
               break;
             case 4:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 6 );
               assert (reader.getFirstByteOffset() == 24 );
               break;
             case 5:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 7 );
               assert (reader.getFirstByteOffset() == 28 );
               break;
             case 6:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 8 );
               assert (reader.getFirstByteOffset() == 32 );
               break;
             case 7:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 9 );
               assert (reader.getFirstByteOffset() == 36 );
               break;
             case 8:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 10 );
               assert (reader.getFirstByteOffset() == 40 );
               break;
             case 9:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 11 );
               assert (reader.getFirstByteOffset() == 44 );
               break;
          }
          break;
      case 11:        // 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
          switch (id) {
             case 0:
               assert( reader.getChunkSize() == 2 );
               assert (reader.getFirstItemOffset() == 0 );
               assert (reader.getFirstByteOffset() == 0 );
               break;
             case 1:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 2 );
               assert (reader.getFirstByteOffset() == 8 );
               break;
             case 2:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 3 );
               assert (reader.getFirstByteOffset() == 12 );
               break;
             case 3:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 4 );
               assert (reader.getFirstByteOffset() == 16 );
               break;
             case 4:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 5 );
               assert (reader.getFirstByteOffset() == 20 );
               break;
             case 5:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 6 );
               assert (reader.getFirstByteOffset() == 24 );
               break;
             case 6:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 7 );
               assert (reader.getFirstByteOffset() == 28 );
               break;
             case 7:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 8 );
               assert (reader.getFirstByteOffset() == 32 );
               break;
             case 8:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 9 );
               assert (reader.getFirstByteOffset() == 36 );
               break;
             case 9:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 10 );
               assert (reader.getFirstByteOffset() == 40 );
               break;
             case 10:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 11 );
               assert (reader.getFirstByteOffset() == 44 );
               break;
          }
          break;
      case 12:        // 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
          switch (id) {
             case 0:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 0 );
               assert (reader.getFirstByteOffset() == 0 );
               break;
             case 1:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 1 );
               assert (reader.getFirstByteOffset() == 4 );
               break;
             case 2:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 2 );
               assert (reader.getFirstByteOffset() == 8 );
               break;
             case 3:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 3 );
               assert (reader.getFirstByteOffset() == 12 );
               break;
             case 4:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 4 );
               assert (reader.getFirstByteOffset() == 16 );
               break;
             case 5:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 5 );
               assert (reader.getFirstByteOffset() == 20 );
               break;
             case 6:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 6 );
               assert (reader.getFirstByteOffset() == 24 );
               break;
             case 7:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 7 );
               assert (reader.getFirstByteOffset() == 28 );
               break;
             case 8:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 8 );
               assert (reader.getFirstByteOffset() == 32 );
               break;
             case 9:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 9 );
               assert (reader.getFirstByteOffset() == 36 );
               break;
             case 10:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 10 );
               assert (reader.getFirstByteOffset() == 40 );
               break;
             case 11:
               assert( reader.getChunkSize() == 1 );
               assert (reader.getFirstItemOffset() == 11 );
               assert (reader.getFirstByteOffset() == 44 );
               break;
          }
          break;
      default:
           cerr << "\n*** Switch: flow should never reach here\n\n";
           MPI_Abort(MPI_COMM_WORLD, 1);
   }

   
   if (id == MASTER) cout << " Passed! " << endl;

}


template <typename ReaderType>
void IntReaderTester::
runReadTests(ReaderType &reader) {
   
   if (id == MASTER) cout << "- Running read() tests... " << flush;

   vector<int> v1 = reader.readChunk();

   assert( reader.getFileSize() == 48 );         // 12 x 4
   assert( reader.getNumItemsInFile() == 12 );
   const int SIZE = reader.getNumItemsInFile();
   vector<int> v2;
   int iVal;
   ifstream fin("files/12ints.txt");
   assert( fin.is_open() );
   for (int i = 0; i < SIZE; ++i) {
      fin >> iVal;
      v2.push_back(iVal);
   } 
   fin.close();

   long start = -1, stop = -1;
   getChunkStartStopValues(id, numProcs, SIZE, start, stop);
// printf("\n\nv1.size(): %lu, (stop-start): %ld\n\n", v1.size(), (stop-start));
   assert( v1.size() == (stop - start) );
   int j = start;
   for (int i = 0; i < v1.size(); ++i) {
      assert( v1[i] == v2[j] );
//        printf("Process %d: v1[%d]: %lf; v2[%d]: %lf\n",
//                      id, i, v1[i], i, v2[i]);
      ++j;
   }

   
   if (id == MASTER) cout << " Passed!" << endl;
}

