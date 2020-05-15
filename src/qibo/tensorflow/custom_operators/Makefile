CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

SRCS = $(wildcard cc/kernels/*.cc) $(wildcard cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

TARGET_LIB = python/ops/_zero_out_ops.so

# zero_out op for CPU
all: $(TARGET_LIB)

$(TARGET_LIB): $(SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

zero_out_test: python/ops/zero_out_ops_test.py $(TARGET_LIB)
	$(PYTHON_BIN_PATH) python/ops/zero_out_ops_test.py

clean:
	rm -f $(ZERO_OUT_TARGET_LIB)
