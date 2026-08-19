# genTextAndBinaryFiles

The files in this folder each generate text- and binary-formatted files, but for different data types:
- *genChars.cpp* generates a random char dataset
- *genDoubles.cpp* generates a random double dataset
- *genInts.cpp* generates a random int dataset

The provided *Makefile* should build all three programs; just download this folder and enter `make`.

Once built, a command such as:

    ./genDoubles 1000000 1M_doubles

will randomly generate a dataset of 1 million double values and write it to two files:
- *1M_doubles.bin*, a binary file containing the dataset in binary format.
- *1M_doubles.txt*, a text file containing 1000000 on its first line, followed by the dataset in text format, with each data value on a separate line.

These three programs are basically the same program with 
1. a `typedef` changed to set ItemType to the desired data type, 
2. constants `LOW` and `HI` (bounds for the range of random values) set appropriately for the data type, and
3. format-strings tweaked as needed for that data type.

To generate a random dataset of a different data type, use any of these programs as a model.
