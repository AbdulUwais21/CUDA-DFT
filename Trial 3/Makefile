FILE 		= Trial3

INCLUDE 	= 	-I./ \
				-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" \
				-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include\helper" \
				-I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include"

LIB 		= -L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\lib\x64" -l cuda -l cudart -l cudart_static
CU_FLAG 	=  -O2 -arch=sm_30 -cudart static -g -G  --ptxas-options=-v
ADDITION	=	--resource-usage
				

.PHONY: clean

%: %.cu
	nvcc $(INCLUDE) $(LIB) $(CU_FLAG) -o $(FILE) $<
txt: $(FILE).exe
	./$(FILE) > output.txt
clean: 
	rm -f *.exe
