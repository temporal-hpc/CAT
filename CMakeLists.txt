cmake_minimum_required(VERSION 3.12)
project(TensorCA)

# Add your source files here
set(SOURCES
    src/main.cpp
    src/foo.cpp
    src/bar.cpp
)

# Add your header files here
set(HEADERS
    include/foo.h
    include/bar.h
)

# Set the include directories
include_directories(include)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})