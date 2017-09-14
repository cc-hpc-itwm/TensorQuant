TF_INC=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#g++ -std=c++11 -shared $1.cc -o $1.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0

CC=g++
CFLAGS=-std=c++11 -shared -fPIC -I $(TF_INC) -O2 -D_GLIBCXX_USE_CXX11_ABI=0
DEPS=
OBJ=$(shell find . -name '*.cc' | sed 's/.cc/.so/g' | sed 's|./||g')


all: $(OBJ)

%.so: %.cc $(DEPS)
	$(CC) $< -o $@ $(CFLAGS)

clean: 
	rm -f *.so

.PHONY: all run clean nodes
