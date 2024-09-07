cmake_minimum_required(VERSION 3.19)
project(CAT LANGUAGES CUDA CXX)
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()


# Collect source and header files
file(GLOB SOURCES "src/*.cpp" "src/*.cu")
file(GLOB HEADERS "include/*.h" "include/*.cuh")

# Set include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

set(NVCCLIBS "-lcudart")


add_library(CAT STATIC ${SOURCES} ${HEADERS})

set_target_properties(CAT PROPERTIES
    CUDA_ARCHITECTURES 70 80 90
)
set_target_properties(CAT PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
# Specify the CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 70 80 90)

find_package(CUDAToolkit REQUIRED)
# Define macros
target_compile_definitions(CAT PRIVATE SMIN=2 SMAX=3 BMIN=3 BMAX=3)


# Create the test executable
add_executable(test_exe tests/test.cu)
target_link_libraries(test_exe CAT ${NVCCLIBS})
target_link_libraries(test_exe CAT ${CUDA_LIBRARIES} ${NVCCLIBS})

# Install targets
install(TARGETS CAT DESTINATION lib/temporal)
install(FILES ${HEADERS} DESTINATION include/temporal)

# Set policy to use new CUDA architectures handling
cmake_policy(SET CMP0104 NEW)
