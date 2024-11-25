# Compiler and linker
NVCC = nvcc

# Compiler flags
CXXFLAGS = -I/usr/local/include 
LDFLAGS = -L/usr/local/lib -lCAT

# Source files
SRCS = main.cu
OBJS = $(SRCS:.cu=.o)

# Output executable
TARGET = main

# Default target
all: $(TARGET)

# Link the object files to create the executable
$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) $(LDFLAGS) -o $(TARGET)

# Compile the source files into object files
%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean