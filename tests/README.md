# tests

This folder contains two dual-purpose C++ programs that (i) test, and (ii) serve as examples for using OO_MPI_IO:
- *readerTester.cpp* tests `ParallelReader`, using the `CharReaderTester`, `IntReaderTester`, and `DoubleReaderTester` classes; and
- *writerTester.cpp* tests `ParallelWriter`, using the `CharWriterTester` and `DoubleWriterTester` classes.

The provided *Makefile* should build both programs. 

The *files* folder contains text and binary files used for testing. 
The binary files were created using MacOS; if you are using a different system, 
you will likely need to recreate the binary files for your system.
See *../genTextAndBinaryFiles* for how to do this.
