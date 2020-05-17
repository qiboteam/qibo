CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

SRCS = $(wildcard cc/kernels/*.cc) $(wildcard cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

TARGET_LIB = python/ops/_qibo_tf_custom_operators.so

ifeq (($shell which nvcc),)
TARGETS = cpuoperator
else
TARGETS = cpuoperator
endif

all: $(TARGETS)

cpuoperator: $(SRCS)
	@echo "Compiling CPU operator"
	$(CXX) $(CFLAGS) -o $(TARGET_LIB) $^ ${LDFLAGS}

cpugpuoperator:
	@echo "cpugpuoperator"

clean:
	rm -f $(TARGET_LIB)

