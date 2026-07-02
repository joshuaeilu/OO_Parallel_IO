/* CharReaderTester.h declares the class that tests Reader
 *   using char values.
 *
 * @author: Joel C. Adams, Calvin University, Fall 2023
 */

#include <iostream>       // cout, ...
#include <fstream>        // ifstream, ofstream, fstream
#include <mpi.h>          // MPI types
#include "../OO_IO.h" // Reader
using namespace std;

class CharReaderTester
{
public:
  CharReaderTester();
  template <typename ReaderType>
  void runTests(ReaderType &reader);
  template <typename ReaderType>
  void runFileTests(const ReaderType &reader);
  template <typename ReaderType>
  void runReadTests(ReaderType &reader);
  template <typename ReaderType>
  void runChunkTests(const ReaderType &reader);
  template <typename ReaderType>
  void runFileTests1(const ReaderType &reader, unsigned numExtras);
  template <typename ReaderType>
  void runReadTests1( ReaderType &reader, unsigned numExtras);
  template <typename ReaderType>
  void runChunkTests1(const ReaderType &reader, unsigned numExtras);

private:
  const int MASTER = 0;
  int id;
  int numProcs;
};

CharReaderTester::CharReaderTester()
{

}

template <typename ReaderType>
void CharReaderTester::runTests(ReaderType & reader)
{
   // set id and getnumpes to support both mpi and mpi
   id = reader.getID();
   numProcs = reader.getNumPEs();

  if (id == MASTER)
    cout << "\nTesting Reader using chars...\n"
         << flush;

  runFileTests(reader);
  runReadTests(reader);
  runChunkTests(reader);

  runFileTests1(reader, 1);
  runReadTests1(reader, 1);
  runChunkTests1(reader, 1);

  runFileTests1(reader, 2);
  runReadTests1(reader, 2);
  runChunkTests1(reader, 2);

  if (id == MASTER)
    cout << "All char tests passed!\n"
         << endl;
}

template <typename ReaderType>
void CharReaderTester::
    runFileTests(const ReaderType &reader)
{
  if (id == MASTER)
    cout << "- Running getter tests..." << flush;

  assert(reader.getID() == id);
  assert(reader.getNumPEs() == numProcs);
  assert(reader.getFileName() == "./files/6chars.bin");
  assert(reader.getItemSize() == 1); // 1 byte in a char

  if (id == MASTER)
    cout << " Passed! " << endl;
}

template <typename ReaderType>
void CharReaderTester::
    runChunkTests(const ReaderType &reader)
{
  if (id == MASTER)
    cout << "- Running chunk-related tests... " << flush;

  switch (numProcs)
  {
  case 1:
    assert(reader.getChunkSize() == 6);
    assert(reader.getFirstItemOffset() == 0);
    assert(reader.getFirstByteOffset() == 0);
    break;
  case 2:
    switch (id)
    {
    case 0:
      assert(reader.getChunkSize() == 3);
      assert(reader.getFirstItemOffset() == 0);
      assert(reader.getFirstByteOffset() == 0);
      break;
    case 1:
      assert(reader.getChunkSize() == 3);
      assert(reader.getFirstItemOffset() == 3);
      assert(reader.getFirstByteOffset() == 3);
      break;
    }
    break;
  case 3:
    switch (id)
    {
    case 0:
      assert(reader.getChunkSize() == 2);
      assert(reader.getFirstItemOffset() == 0);
      assert(reader.getFirstByteOffset() == 0);
      break;
    case 1:
      assert(reader.getChunkSize() == 2);
      assert(reader.getFirstItemOffset() == 2);
      assert(reader.getFirstByteOffset() == 2);
      break;
    case 2:
      assert(reader.getChunkSize() == 2);
      assert(reader.getFirstItemOffset() == 4);
      assert(reader.getFirstByteOffset() == 4);
      break;
    }
    break;
  case 4:
    switch (id)
    {
    case 0:
      assert(reader.getChunkSize() == 2);
      assert(reader.getFirstItemOffset() == 0);
      assert(reader.getFirstByteOffset() == 0);
      break;
    case 1:
      assert(reader.getChunkSize() == 2);
      assert(reader.getFirstItemOffset() == 2);
      assert(reader.getFirstByteOffset() == 2);
      break;
    case 2:
      assert(reader.getChunkSize() == 1);
      assert(reader.getFirstItemOffset() == 4);
      assert(reader.getFirstByteOffset() == 4);
      break;
    case 3:
      assert(reader.getChunkSize() == 1);
      assert(reader.getFirstItemOffset() == 5);
      assert(reader.getFirstByteOffset() == 5);
      break;
    }
    break;
  case 5:
    switch (id)
    {
    case 0:
      assert(reader.getChunkSize() == 2);
      assert(reader.getFirstItemOffset() == 0);
      assert(reader.getFirstByteOffset() == 0);
      break;
    case 1:
      assert(reader.getChunkSize() == 1);
      assert(reader.getFirstItemOffset() == 2);
      assert(reader.getFirstByteOffset() == 2);
      break;
    case 2:
      assert(reader.getChunkSize() == 1);
      assert(reader.getFirstItemOffset() == 3);
      assert(reader.getFirstByteOffset() == 3);
      break;
    case 3:
      assert(reader.getChunkSize() == 1);
      assert(reader.getFirstItemOffset() == 4);
      assert(reader.getFirstByteOffset() == 4);
      break;
    case 4:
      assert(reader.getChunkSize() == 1);
      assert(reader.getFirstItemOffset() == 5);
      assert(reader.getFirstByteOffset() == 5);
      break;
    default:
      cerr << "\n*** Switch 2: flow should not reach here\n\n";
      
    }
    break;
  default:
    cerr << "\n*** Switch: flow should never reach here\n\n";
    
  }

  if (id == MASTER)
    cout << " Passed! " << endl;
}

template <typename ReaderType>
void CharReaderTester::
    runReadTests(ReaderType &reader)
{
  if (id == MASTER)
    cout << "- Running read() tests... " << flush;

  std::span<const char> v1 = reader.readChunk();

  assert(reader.getFileSize() == 6); // 1 x 6
  printf("NumItemsInFile: %ld\n", reader.getNumItemsInFile());
  assert(reader.getNumItemsInFile() == 6);
  const int SIZE = reader.getNumItemsInFile();
  vector<char> v2;
  char cVal;
  ifstream fin("files/6chars.txt");
  assert(fin.is_open());
  for (int i = 0; i < SIZE; ++i)
  {
    fin >> cVal;
    v2.push_back(cVal);
  }
  fin.close();

  long start = -1, stop = -1;
  getChunkStartStopValues(id, numProcs, SIZE, start, stop);
  // printf("\n\nv1.size(): %lu, (stop-start): %ld\n\n", v1.size(), (stop-start));
  assert(v1.size() == (stop - start));
  int j = start;
  for (int i = 0; i < v1.size(); ++i)
  {
    assert(v1[i] == v2[j]);
    //        printf("Process %d: v1[%d]: %lf; v2[%d]: %lf\n",
    //                      id, i, v1[i], i, v2[i]);
    ++j;
  }

  if (id == MASTER)
    cout << " Passed (6 chars)!" << endl;
}

template <typename ReaderType>
void CharReaderTester::
    runFileTests1(const ReaderType &reader, unsigned numExtras)
{
  if (id == MASTER)
    cout << "- Running getter tests..." << flush;

  assert(reader.getID() == id);
  assert(reader.getNumPEs() == numProcs);
  assert(reader.getFileName() == "./files/6chars.bin");
  assert(reader.getItemSize() == 1); // 1 byte in a char

  if (id == MASTER)
    cout << " Passed! " << endl;
}

template <typename ReaderType>
void CharReaderTester::
    runChunkTests1(const ReaderType &reader, unsigned numExtras)
{
  if (id == MASTER)
    cout << "- Running chunk-related tests... " << flush;

  switch (numProcs)
  {
  case 1:
    assert(reader.getChunkSize() == 6);
    assert(reader.getFirstItemOffset() == 0);
    assert(reader.getFirstByteOffset() == 0);
    break;
  case 2:
    switch (id)
    {
    case 0:
      assert(reader.getChunkSize() == 3 + numExtras);
      assert(reader.getFirstItemOffset() == 0);
      assert(reader.getFirstByteOffset() == 0);
      break;
    case 1:
      assert(reader.getChunkSize() == 3);
      assert(reader.getFirstItemOffset() == 3);
      assert(reader.getFirstByteOffset() == 3);
      break;
    }
    break;
  case 3:
    switch (id)
    {
    case 0:
      assert(reader.getChunkSize() == 2 + numExtras);
      assert(reader.getFirstItemOffset() == 0);
      assert(reader.getFirstByteOffset() == 0);
      break;
    case 1:
      assert(reader.getChunkSize() == 2 + numExtras);
      assert(reader.getFirstItemOffset() == 2);
      assert(reader.getFirstByteOffset() == 2);
      break;
    case 2:
      assert(reader.getChunkSize() == 2);
      assert(reader.getFirstItemOffset() == 4);
      assert(reader.getFirstByteOffset() == 4);
      break;
    }
    break;
  case 4:
    switch (id)
    {
    case 0:
      assert(reader.getChunkSize() == 2 + numExtras);
      assert(reader.getFirstItemOffset() == 0);
      assert(reader.getFirstByteOffset() == 0);
      break;
    case 1:
      assert(reader.getChunkSize() == 2 + numExtras);
      assert(reader.getFirstItemOffset() == 2);
      assert(reader.getFirstByteOffset() == 2);
      break;
    case 2:
      assert(reader.getChunkSize() == 1 + 1);
      assert(reader.getFirstItemOffset() == 4);
      assert(reader.getFirstByteOffset() == 4);
      break;
    case 3:
      assert(reader.getChunkSize() == 1);
      assert(reader.getFirstItemOffset() == 5);
      assert(reader.getFirstByteOffset() == 5);
      break;
    }
    break;
  case 5:
    switch (id)
    {
    case 0:
      assert(reader.getChunkSize() == 2 + numExtras);
      assert(reader.getFirstItemOffset() == 0);
      assert(reader.getFirstByteOffset() == 0);
      break;
    case 1:
      assert(reader.getChunkSize() == 1 + numExtras);
      assert(reader.getFirstItemOffset() == 2);
      assert(reader.getFirstByteOffset() == 2);
      break;
    case 2:
      assert(reader.getChunkSize() == 1 + numExtras);
      assert(reader.getFirstItemOffset() == 3);
      assert(reader.getFirstByteOffset() == 3);
      break;
    case 3:
      assert(reader.getChunkSize() == 1 + 1);
      assert(reader.getFirstItemOffset() == 4);
      assert(reader.getFirstByteOffset() == 4);
      break;
    case 4:
      assert(reader.getChunkSize() == 1);
      assert(reader.getFirstItemOffset() == 5);
      assert(reader.getFirstByteOffset() == 5);
      break;
    default:
      cerr << "\n*** Switch 2: flow should not reach here\n\n";
      
    }
    break;
  default:
    cerr << "\n*** Switch: flow should never reach here\n\n";
    
  }

  if (id == MASTER)
    cout << " Passed! " << endl;
}


template <typename ReaderType>
void CharReaderTester::
    runReadTests1( ReaderType &reader, unsigned numExtras)
{
  if (id == MASTER)
    cout << "- Running read() tests... " << flush;

  std::span<const char> v1 = reader.readChunkPlus(numExtras);

  assert(reader.getFileSize() == 6); // 1 x 6
  printf("NumItemsInFile: %ld\n", reader.getNumItemsInFile());
  fflush(stdout);
  assert(reader.getNumItemsInFile() == 6);
  const int SIZE = reader.getNumItemsInFile();
  vector<char> v2;
  char cVal;
  ifstream fin("files/6chars.txt");
  assert(fin.is_open());
  for (int i = 0; i < SIZE; ++i)
  {
    fin >> cVal;
    v2.push_back(cVal);
  }
  fin.close();

  long start = -1, stop = -1;
  getChunkStartStopValues(id, numProcs, SIZE, start, stop);
  if (id < numProcs - 1)
  {
    stop += numExtras;
  }
  if (stop > SIZE)
  {
    stop = SIZE;
  }

  // printf("\n\nv1.size(): %lu, (stop-start): %ld\n\n", v1.size(), (stop-start));
  assert(v1.size() == (stop - start));
  int j = start;
  for (int i = 0; i < v1.size(); ++i)
  {
    // std::cout << v1[i] << " : " << v2[j] << std::endl;
    assert(v1[i] == v2[j]);
    //        printf("Process %d: v1[%d]: %lf; v2[%d]: %lf\n",
    //                      id, i, v1[i], i, v2[i]);
    ++j;
  }

  if (id == MASTER)
    cout << " Passed (6 chars, "
         << numExtras << " extras)!" << endl;
}
