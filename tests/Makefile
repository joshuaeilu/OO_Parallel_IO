PROG1  = readerTester
PROG2  = writerTester
SRC1   = $(PROG1).cpp
SRC2   = $(PROG2).cpp
INCL1  = ../OO_MPI_IO.h \
          DoubleReaderTester.h \
          IntReaderTester.h \
          CharReaderTester.h
INCL2  = ../OO_MPI_IO.h \
          DoubleWriterTester.h \
          CharWriterTester.h

SHELL  = /bin/bash

CC     = mpicxx
CFLAGS = -Wall -ansi -std=c++11

OS     = $(shell uname -s)

ifeq ($(OS), Darwin)
CFLAGS += -Xclang -fopenmp
LFLAGS += -lomp
else
CFLAGS += -pedantic
LFLAGS += -fopenmp
endif

LFLAGS1 = $(LFLAGS) -o $(PROG1) 
LFLAGS2 = $(LFLAGS) -o $(PROG2) 

all: $(PROG1) $(PROG2)

$(PROG1): $(SRC1) $(INCL1)
	$(CC) $(CFLAGS) $(SRC1) $(LFLAGS1)

$(PROG2): $(SRC2) $(INCL2)
	$(CC) $(CFLAGS) $(SRC2) $(LFLAGS2)

clean:
	rm -f $(PROG1) $(PROG2) a.out *~ *# *.o

