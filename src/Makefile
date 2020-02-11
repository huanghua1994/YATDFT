CC  = icc
LIB = libYATDFT.a

USE_MKL      = 1
USE_OPENBLAS = 0
USE_LIBXC    = 0

SIMINT_INSTALL_DIR   = /home/mkurisu/Workspace/simint/build-avx/install
LIBXC_INSTALL_DIR    = /home/mkurisu/Workspace/libxc/install
OPENBLAS_INSTALL_DIR = /home/mkurisu/Workspace/OpenBLAS-0.3.8/install

DEFS    = -D__ALIGNLEN__=64
INCS    = -I./include -I$(SIMINT_INSTALL_DIR)/include
CFLAGS  = -Wall -g -O3 -std=gnu99 $(DEFS)

ifeq ($(CC), icc)
AR      = xiar rcs
CFLAGS += -qopenmp -xHost
endif

ifeq ($(CC), gcc)
AR      = ar rcs
CFLAGS += -fopenmp -march=native -Wno-unused-result -Wno-unused-function
endif

ifeq ($(USE_MKL), 1)
DEFS   += -DUSE_MKL
CFLAGS += -mkl
endif

ifeq ($(USE_OPENBLAS), 1)
DEFS   += -DUSE_OPENBLAS
INCS   += -I$(OPENBLAS_INSTALL_DIR)/include
endif

ifeq ($(USE_LIBXC), 1)
INCS   += -I$(LIBXC_INSTALL_DIR)/include
DEFS   += -DUSE_LIBXC
endif

OBJS =  CMS_basis.o CMS_simint.o utils.o ket_sp_list.o TinyDFT_typedef.o \
        acc_JKmat.o build_HF_mat.o gen_Lebedev_grid.o gen_int_grid.o     \
        eval_XC_func.o build_XCmat.o CDIIS.o setup_DF.o build_JKmat_DF.o \
        build_Dmat.o 

$(LIB): $(OBJS) 
	$(AR) $@ $^

CMS_basis.o: Makefile ./include/*.h CMS_basis.c 
	$(CC) $(CFLAGS) $(INCS) -c CMS_basis.c -o $@ 

CMS_simint.o: Makefile ./include/*.h CMS_simint.c 
	$(CC) $(CFLAGS) $(INCS) -c CMS_simint.c -o $@ 

utils.o: Makefile ./include/*.h utils.c 
	$(CC) $(CFLAGS) $(INCS) -c utils.c -o $@ 

ket_sp_list.o: Makefile ./include/*.h ket_sp_list.c 
	$(CC) $(CFLAGS) $(INCS) -c ket_sp_list.c -o $@ 

TinyDFT_typedef.o: Makefile ./include/*.h TinyDFT_typedef.c 
	$(CC) $(CFLAGS) $(INCS) -c TinyDFT_typedef.c -o $@ 

acc_JKmat.o: Makefile ./include/*.h acc_JKmat.c 
	$(CC) $(CFLAGS) $(INCS) -c acc_JKmat.c -o $@ 

build_HF_mat.o: Makefile ./include/*.h build_HF_mat.c 
	$(CC) $(CFLAGS) $(INCS) -c build_HF_mat.c -o $@ 

gen_Lebedev_grid.o: Makefile ./include/*.h gen_Lebedev_grid.c 
	$(CC) $(CFLAGS) $(INCS) -c gen_Lebedev_grid.c -o $@ 

gen_int_grid.o: Makefile ./include/*.h gen_int_grid.c 
	$(CC) $(CFLAGS) $(INCS) -c gen_int_grid.c -o $@ 

eval_XC_func.o: Makefile ./include/*.h eval_XC_func.c 
	$(CC) $(CFLAGS) $(INCS) -c eval_XC_func.c -o $@ 

build_XCmat.o: Makefile ./include/*.h build_XCmat.c 
	$(CC) $(CFLAGS) $(INCS) -c build_XCmat.c -o $@ 

CDIIS.o: Makefile ./include/*.h CDIIS.c 
	$(CC) $(CFLAGS) $(INCS) -c CDIIS.c -o $@ 

setup_DF.o: Makefile ./include/*.h setup_DF.c 
	$(CC) $(CFLAGS) $(INCS) -c setup_DF.c -o $@ 

build_JKmat_DF.o: Makefile ./include/*.h build_JKmat_DF.c 
	$(CC) $(CFLAGS) $(INCS) -c build_JKmat_DF.c -o $@ 

build_Dmat.o: Makefile ./include/*.h build_Dmat.c 
	$(CC) $(CFLAGS) $(INCS) -c build_Dmat.c -o $@ 

clean:
	rm -f *.o $(LIB)
