g++ -std=c++11 -shared $1.cc -o $1.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
