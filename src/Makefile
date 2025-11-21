
NVCC        = nvcc
NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_75,code=\"sm_75,compute_75\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O2
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = PTOSS
OBJ	        = PTOSS.o PTOSS_kernel.o Board.o

default: $(EXE)

PTOSS.o : PTOSS.cu PTOSS_kernel.cu PTOSS_kernel.h Board.cu Board.h
	$(NVCC) -c -o $@ PTOSS.cu $(NVCC_FLAGS)

PTOSS_kernel.o: PTOSS_kernel.cu PTOSS_kernel.h Board.cu Board.h
	$(NVCC) -c -o $@ PTOSS_kernel.cu $(NVCC_FLAGS)

Board.o: Board.cu Board.h
	$(NVCC) -c -o $@ Board.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
