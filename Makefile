# Makefile,
CC := g++ -std=c++17
NVCC := nvcc
# path macros
BIN_PATH := bin
OBJ_PATH := obj
DBG_PATH := debug
SRC_PATH := src

# Compiler Macros used in code
BSIZE3DX=32
BSIZE3DY=32
BSIZE3DZ=1
DP := NO
MEASURE_POWER := NO

NREGIONS_H := 6
NREGIONS_V := 8
# Compiler macros used on TC based automatas

TARGET_NAME := prog

TARGET := $(BIN_PATH)/$(TARGET_NAME)
TARGET_DEBUG := $(DBG_PATH)/$(TARGET_NAME)

NVCCLIBS := -lnvidia-ml
DPFLAGS := -rdc=true -lcudadevrt -DDP
ARCH=-arch sm_86

CCOBJFLAGS=-O3
CUOBJFLAGS=-O3

GLOBALDEFINES := -DBSIZE3DX=${BSIZE3DX} -DBSIZE3DY=${BSIZE3DY} -DBSIZE3DZ=${BSIZE3DZ} -D${MEASURE_POWER} -DNREGIONS_H=${NREGIONS_H} -DNREGIONS_V=${NREGIONS_V}
CCDEFINES :=
CUDEFINES :=
DBGDEFINES := -DDEBUG -DVERIFY

NVCCFLAGS=${ARCH} ${NVCCLIBS}

ifneq (${DP}, NO)
	NVCCFLAGS := ${NVCCFLAGS} ${DPFLAGS}
endif

CPP_SRC := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.cpp)))
CUDA_SRC := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.cu)))

OBJ := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(CPP_SRC)))))
CUDA_OBJ := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(CUDA_SRC)))))

DBG_OBJ := $(addprefix $(DBG_PATH)/, $(addsuffix .o, $(notdir $(basename $(CPP_SRC)))))
DBG_CUDA_OBJ := $(addprefix $(DBG_PATH)/, $(addsuffix .o, $(notdir $(basename $(CUDA_SRC)))))

default: makedir all

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp
	$(CC) $(CCDEFINES) $(CCOBJFLAGS) $(GLOBALDEFINES) -c -o $@ $<

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cu
	$(NVCC) $(NVCCFLAGS) $(CUOBJFLAGS) $(GLOBALDEFINES) -MMD -c -o $@ $<


$(DBG_PATH)/%.o: $(SRC_PATH)/%.cpp
	$(CC) $(CCDEFINES) $(CCOBJFLAGS) $(GLOBALDEFINES) $(DBGDEFINES) -c -o $@ $<

$(DBG_PATH)/%.o: $(SRC_PATH)/%.cu
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
	-@rm $(DBG_PATH)/*
	-@rm $(OBJ_PATH)/*
	-@rm $(TARGET)
	-@rm $(TARGET_DEBUG)

-include $(OBJ_PATH)/*.d
-include $(DBG_PATH)/*.d

