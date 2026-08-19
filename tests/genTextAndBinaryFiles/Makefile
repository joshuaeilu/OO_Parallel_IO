PROGS        := genDoubles genInts genChars
CXX          := g++
CXXFLAGS     := -std=c++11 -pedantic -Wall -Wextra

ITEM_COUNTS  := 1 3 4 5 8 9

# Assumes this generator folder is inside the tests folder,
# next to tests/files.
OUTPUT_DIR   := ../files

.PHONY: all programs test-data clean clean-data

all: programs test-data

programs: $(PROGS)

$(PROGS): %: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

test-data: programs
	mkdir -p $(OUTPUT_DIR)
	@set -e; \
	for count in $(ITEM_COUNTS); do \
		echo "Generating $${count} doubles..."; \
		./genDoubles $$count "$(OUTPUT_DIR)/$${count}doubles"; \
		echo "Generating $${count} ints..."; \
		./genInts $$count "$(OUTPUT_DIR)/$${count}ints"; \
		echo "Generating $${count} chars..."; \
		./genChars $$count "$(OUTPUT_DIR)/$${count}chars"; \
	done

clean:
	rm -f $(PROGS) *.o *~ *#

clean-data:
	@for count in $(ITEM_COUNTS); do \
		rm -f "$(OUTPUT_DIR)/$${count}doubles.bin"; \
		rm -f "$(OUTPUT_DIR)/$${count}doubles.txt"; \
		rm -f "$(OUTPUT_DIR)/$${count}ints.bin"; \
		rm -f "$(OUTPUT_DIR)/$${count}ints.txt"; \
		rm -f "$(OUTPUT_DIR)/$${count}chars.bin"; \
		rm -f "$(OUTPUT_DIR)/$${count}chars.txt"; \
	done