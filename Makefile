all: build/lib.clang.min.hpp build/lib.gcc.min.hpp

pack: pack.cpp
	$(CXX) $< -o $@ -std=c++17 -O2

build/lib.clang.min.hpp: lib.hpp pack
	python3 pack.py $< clang | ./pack >$@

build/lib.gcc.min.hpp: lib.hpp pack
	python3 pack.py $< gcc | ./pack >$@
