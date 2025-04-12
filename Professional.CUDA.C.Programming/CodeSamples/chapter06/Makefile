CU_APPS=asyncAPI simpleCallback simpleHyperqBreadth simpleHyperqDependence \
        simpleHyperqDepth simpleHyperqOpenmp simpleMultiAddBreadth \
        simpleMultiAddDepth simpleHyperq
C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -g -G -Xcompiler -fopenmp -O0 -Xptxas -O0 -lineinfo -O0 -o $@ $< -lgomp
#	nvcc -O2 -Xcompiler -fopenmp -o $@ $< -lgomp
#	nvcc -O2 -arch=sm_20 -Xcompiler -fopenmp -o $@ $< -lgomp
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
