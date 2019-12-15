stencil: stencil.c
	mpiicc \
	-O3 -xHost -std=c99 -qopenmp \
	stencil.c -o stencil
