PROG1     = genDoubles
PROG2     = genInts
PROG3     = genChars
SRC1      = $(PROG1).cpp
SRC2      = $(PROG2).cpp
SRC3      = $(PROG3).cpp
CC        = g++
CFLAGS    = -ansi -pedantic -std=c++11
LFLAGS1   = -o $(PROG1)
LFLAGS2   = -o $(PROG2)
LFLAGS3   = -o $(PROG3)

all: $(PROG1) $(PROG2) $(PROG3)

$(PROG1): $(SRC1)
	$(CC) $(CFLAGS) $(SRC1) $(LFLAGS1)

$(PROG2): $(SRC2)
	$(CC) $(CFLAGS) $(SRC2) $(LFLAGS2)

$(PROG3): $(SRC3)
	$(CC) $(CFLAGS) $(SRC3) $(LFLAGS3)

clean:
	rm -f $(PROG1) $(PROG2) $(PROG3) *.o *~ *#

