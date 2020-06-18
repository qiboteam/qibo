# TARGET LIBRARIES
TARGET_LIB = python/ops/_qibo_tf_custom_operators.so
TARGET_LIB_CUDA = python/ops/_qibo_tf_custom_operators_cuda.so

TARGETS = $(TARGET_LIB)

ifneq ($(CUDA_PATH),)
	TARGETS += $(TARGET_LIB_CUDA)
endif

SRCS = $(wildcard cc/kernels/*.cc) $(wildcard cc/ops/*.cc)
CUDASRC = $(wildcard cc/kernels/*.cu.cc)

CXX := g++
NVCC := $(CUDA_PATH)/bin/nvcc
PYTHON_BIN_PATH = python

SOURCES = $(filter-out $(CUDASRC), $(SRCS))

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
CFLAGS_CUDA = $(CFLAGS) -D GOOGLE_CUDA=1 -I$(CUDA_PATH)/include
CFLAGS_NVCC = ${TF_CFLAGS} -O2 -std=c++11 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

LDFLAGS = -shared ${TF_LFLAGS}
LDFLAGS_CUDA = $(LDFLAGS) -L$(CUDA_PATH)/lib64 -lcudart

OBJECT_SRCS = $(SOURCES:.cc=.o)
OBJECT_SRCS_CUDA = $(CUDASRC:.cc=.cudao)

all: $(TARGETS)

$(TARGET_LIB): $(OBJECT_SRCS)
	$(CXX) -o $@ $(CFLAGS) $^ $(LDFLAGS)

$(TARGET_LIB_CUDA): $(OBJECT_SRCS_CUDA)
	$(CXX) -o $@ $(CFLAGS_CUDA) $^ $(LDFLAGS_CUDA)

%.o: %.cc
	$(CXX) -c $(CFLAGS) $^ -o $@

%.cudao: %.cc
	$(CXX) -c $(CFLAGS_CUDA) $^ -o $@

%.cu.cudao: %.cu.cc
	$(NVCC) -c -o $@ $^ $(CFLAGS_NVCC)

clean:
	rm -f $(TARGET_LIB) $(TARGET_LIB_CUDA) $(OBJECT_SRCS) $(OBJECT_SRCS_CUDA)

