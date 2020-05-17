# Modify just these 2 lines
SRCS = $(wildcard cc/kernels/*.cc) $(wildcard cc/ops/*.cc)
CUDASRC = $(wildcard cc/kernels/*.cu.cc)

CXX := g++
PYTHON_BIN_PATH = python

SOURCES = $(filter-out $(CUDASRC), $(SRCS))

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

TARGET_LIB = python/ops/_qibo_tf_custom_operators.so

ifeq ($(CUDA_PATH),)
TARGETS = cpuoperator
else
TARGETS = cpugpuoperator
NVCC := $(CUDA_PATH)/bin/nvcc
CFLAGS += -D GOOGLE_CUDA=1 -I$(CUDA_PATH)/include
LDFLAGS += -L$(CUDA_PATH)/lib -lcudart
CFLAGS_CUDA = ${TF_CFLAGS} -O2 -std=c++11 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr
endif

all: clean $(TARGETS)

OBJECT_SRCS = $(SOURCES:.cc=.o)
OBJECT_CUDASRCS = $(CUDASRC:.cc=.o)

cpuoperator: $(OBJECT_SRCS)
	@echo "CUDA_PATH not set, compiling CPU-only operator"
	$(CXX) -o $(TARGET_LIB) $(CFLAGS) $^ $(LDFLAGS)

cpugpuoperator: $(OBJECT_CUDASRCS) $(OBJECT_SRCS)
	@echo "Compiling GPU operator"
	$(CXX) $(CFLAGS) -o $(TARGET_LIB) $^ $(LDFLAGS)

%.o: %.cc
	$(CXX) -c $(CFLAGS) $^ -o $@

%.cu.o: %.cu.cc
	$(NVCC) -c -o $@ $^ $(CFLAGS_CUDA) $(TF_LFLAGS)

clean:
	rm -f $(TARGET_LIB) $(OBJECT_SRCS) $(OBJECT_CUDASRCS)

