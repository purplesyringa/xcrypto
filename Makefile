all: build/lib.clang.min.hpp build/lib.gcc.min.hpp

pack: pack.cpp
	$(CXX) $< -o $@ -std=c++17 -O2

build/lib.formatted.hpp: lib.hpp
	clang-format-16 --style=file:minimize.clang-format $< >$@

build/lib.clang.min.hpp: build/lib.formatted.hpp pack
	python3 pack.py $< clang | ./pack >$@
build/lib.gcc.min.hpp: build/lib.formatted.hpp pack
	python3 pack.py $< gcc | ./pack >$@
