# Compiler and linker
NVCC = nvcc

# Compiler flags
CXXFLAGS = -I/usr/local/include 
LDFLAGS = -L/usr/local/lib -lCAT

# Source files
MAIN_SRC = main.cu
TEST_CG_SRC = test_CG.cu

# Object files
MAIN_OBJ = $(MAIN_SRC:.cu=.o)
TEST_CG_OBJ = $(TEST_CG_SRC:.cu=.o)

# Output executables
MAIN_TARGET = main
TEST_CG_TARGET = test_CG

# Default target
all: $(MAIN_TARGET) $(TEST_CG_TARGET)

# Link the object files to create the executables
$(MAIN_TARGET): $(MAIN_OBJ)
	$(NVCC) $(MAIN_OBJ) $(LDFLAGS) -o $(MAIN_TARGET)

$(TEST_CG_TARGET): $(TEST_CG_OBJ)
	$(NVCC) $(TEST_CG_OBJ) $(LDFLAGS) -o $(TEST_CG_TARGET)

# Compile the source files into object files
%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(MAIN_OBJ) $(TEST_CG_OBJ) $(MAIN_TARGET) $(TEST_CG_TARGET)

# Phony targets
.PHONY: all clean