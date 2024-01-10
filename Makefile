# Makefile,
CC := g++ -std=c++17
NVCC := nvcc
# path macros
BIN_PATH := bin
OBJ_PATH := obj
DBG_PATH := debug
SRC_PATH := src

# Compiler Macros used in code
BSIZE3DX=16
BSIZE3DY=16
BSIZE3DZ=1
DP := NO
MEASURE_POWER := NO

NREGIONS_H := 2
NREGIONS_V := 5

RADIUS := 1
SMIN := 2
SMAX := 3
BMIN := 3
BMAX := 3

# RADIUS := 4
# SMIN := 40
# SMAX := 80
# BMIN := 41
# BMAX := 81

TARGET_NAME := prog

TARGET := $(BIN_PATH)/$(TARGET_NAME)
TARGET_DEBUG := $(DBG_PATH)/$(TARGET_NAME)

NVCCLIBS := -lnvidia-ml
DPFLAGS := -rdc=true -lcudadevrt -DDP
ARCH=-arch=sm_86

CCOBJFLAGS=-O3
CUOBJFLAGS=-O3

GLOBALDEFINES := -Isrc/ -w -DRADIUS=${RADIUS} -DSMIN=${SMIN} -DSMAX=${SMAX} -DBMIN=${BMIN} -DBMAX=${BMAX} -DBSIZE3DX=${BSIZE3DX} -DBSIZE3DY=${BSIZE3DY} -DBSIZE3DZ=${BSIZE3DZ} -D${MEASURE_POWER} -DNREGIONS_H=${NREGIONS_H} -DNREGIONS_V=${NREGIONS_V}
CCDEFINES :=
CUDEFINES :=
DBGDEFINES := -DDEBUG -DVERIFY

NVCCFLAGS=${ARCH} ${NVCCLIBS}

ifneq (${DP}, NO)
	NVCCFLAGS := ${NVCCFLAGS} ${DPFLAGS}
endif

CPP_SRC := $(shell find $(SRC_PATH) -name "*.cpp")
CUDA_SRC := $(shell find $(SRC_PATH) -name "*.cu")

# Generate the list of .o files with subdirectory structure
OBJ := $(patsubst $(SRC_PATH)/%.cpp, $(OBJ_PATH)/%.o, $(CPP_SRC))
CUDA_OBJ := $(patsubst $(SRC_PATH)/%.cu, $(OBJ_PATH)/%.o, $(CUDA_SRC))

DBG_OBJ := $(patsubst $(SRC_PATH)/%.cpp, $(DBG_PATH)/%.o, $(CPP_SRC))
DBG_CUDA_OBJ := $(patsubst $(SRC_PATH)/%.cu, $(DBG_PATH)/%.o, $(CUDA_SRC))

default: makedir all

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp
	@mkdir -p $(@D)
	$(CC) $(CCDEFINES) $(CCOBJFLAGS) $(GLOBALDEFINES) -c -o $@ $<

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(CUOBJFLAGS) $(GLOBALDEFINES) -MMD -c -o $@ $<


$(DBG_PATH)/%.o: $(SRC_PATH)/%.cpp
	@mkdir -p $(@D)
	$(CC) $(CCDEFINES) $(CCOBJFLAGS) $(GLOBALDEFINES) $(DBGDEFINES) -c -o $@ $<

$(DBG_PATH)/%.o: $(SRC_PATH)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(CUOBJFLAGS) $(GLOBALDEFINES) $(DBGDEFINES) -MMD -c -o $@ $<

$(TARGET): $(OBJ) $(CUDA_OBJ)
	$(NVCC) ${NVCCFLAGS} -o $@ $(OBJ) $(CUDA_OBJ)

$(TARGET_DEBUG) : $(DBG_OBJ) $(DBG_CUDA_OBJ)
	$(NVCC) ${NVCCFLAGS} -o $@ $(DBG_OBJ) $(DBG_CUDA_OBJ)


# phony rules
.PHONY: makedir
makedir:
	@mkdir -p $(BIN_PATH) $(OBJ_PATH) $(DBG_PATH)

.PHONY: debug
debug: $(TARGET_DEBUG)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	-@rm -r $(DBG_PATH)/*
	-@rm -r $(OBJ_PATH)/*
	-@rm $(TARGET)
	-@rm $(TARGET_DEBUG)

-include $(OBJ_PATH)/*.d
-include $(DBG_PATH)/*.d

