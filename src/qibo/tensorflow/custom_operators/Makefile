CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

SRCS = $(wildcard cc/kernels/*.cc) $(wildcard cc/ops/*.cc)
CUDASRC = $(wildcard cc/kernels/*.cu)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
CFLAGS_CUDA = ${TF_CFLAGS} -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

TARGET_LIB = python/ops/_qibo_tf_custom_operators.so

ifeq (($shell which nvcc),)
TARGETS = cpuoperator
else
TARGETS = cpugpuoperator
endif

all: $(TARGETS)

cpuoperator: $(SRCS)
	@echo "Compiling CPU operator"
	$(CXX) $(CFLAGS) -o $(TARGET_LIB) $^ ${LDFLAGS}

cpugpuoperator: $(CUDASRC)
	@echo "Compiling GPU operator"
	$(NVCC) -c -o $^.o $^ $(CFLAGS_CUDA) $(TF_LFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr
	$(CXX) $(CFLAGS) -o $(TARGET_LIB) $^.o $(SRCS) ${LDFLAGS} -D GOOGLE_CUDA=1 -I/opt/cuda-10.1/include -L/opt/cuda-10.1/lib -lcudart

clean:
	rm -f $(TARGET_LIB)

