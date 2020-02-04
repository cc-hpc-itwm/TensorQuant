CC=g++
TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CFLAGS=-std=c++11 -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2

DEPS=
OBJ=$(shell find . -maxdepth 1 -name '*.cc' | sed 's/.cc/.so/g' | sed 's|./||g')


all: $(OBJ)

%.so: %.cc $(DEPS)
	$(CC) $< -o $@ $(CFLAGS)

clean:
	rm -f *.so

.PHONY: all run clean nodes
