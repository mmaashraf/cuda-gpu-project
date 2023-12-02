NVCC        = nvcc
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv)
OPENCV_LIBS = $(shell pkg-config --libs opencv)
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 -arch=sm_20
else
NVCC_FLAGS  = -O3 --std=c++03
endif
LD_FLAGS    = -lcudart
EXE         = sgemm-tiled
OBJ         = main.o

default: $(EXE)

main.o: main.cu
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS) $(OPENCV_CFLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(OPENCV_LIBS)

clean:
	rm -rf *.o $(EXE)
