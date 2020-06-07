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

OBJECT_SRCS = $(SOURCES:.cc=.o)

ifneq ($(CUDA_PATH),)
OBJECT_SRCS += $(CUDASRC:.cc=.o)
NVCC := $(CUDA_PATH)/bin/nvcc
CFLAGS += -D GOOGLE_CUDA=1 -I$(CUDA_PATH)/include
LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart
CFLAGS_CUDA = ${TF_CFLAGS} -O2 -std=c++11 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr
endif

all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJECT_SRCS)
	$(CXX) -o $@ $(CFLAGS) $^ $(LDFLAGS)

%.o: %.cc
	$(CXX) -c $(CFLAGS) $^ -o $@

%.cu.o: %.cu.cc
	$(NVCC) -c -o $@ $^ $(CFLAGS_CUDA)

clean:
	rm -f $(TARGET_LIB) $(OBJECT_SRCS) $(OBJECT_CUDASRCS)

