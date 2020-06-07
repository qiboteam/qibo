CC := gcc
SRCS = $(wildcard c/*.c)
CFLAGS = -fopenmp -fPIC -O2
LDFLAGS = -shared
TARGET_LIB = python/_qibo_custom_functions.so
OBJECT_SRCS = $(SRCS:.c=.o)
ifeq ($(shell uname -s),Darwin)
    CFLAGS = -fPIC -O2
endif

all: clean $(TARGET_LIB)

$(TARGET_LIB): $(OBJECT_SRCS)
	$(CC) -o $(TARGET_LIB) $(CFLAGS) $^ $(LDFLAGS)

%.o: %.c
	$(CC) -c $(CFLAGS) $^ -o $@

clean:
	rm -f $(TARGET_LIB) $(OBJECT_SRCS)

