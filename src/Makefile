CC  = icc
AR  = xiar rcs
LIB = libYATDFT.a

BLAS_LIB       = -mkl=parallel
SIMINT_INCDIR  = /home/mkurisu/Workspace/simint/build-avx/install/include
LIBXC_INCDIR   = /home/mkurisu/Workspace/libxc/install/include

DEFS    = -D__ALIGNLEN__=64
INCS    = -I./ -I$(SIMINT_INCDIR) -I$(LIBXC_INCDIR)

CFLAGS  = -Wall -g -O3 -qopenmp -std=gnu99 -xHost $(DEFS) $(BLAS_LIB)
LDFLAGS = -lpthread -qopenmp $(BLAS_LIB)

OBJS =  CMS_basis.o CMS_simint.o utils.o ket_sp_list.o TinyDFT_typedef.o \
		acc_JKmat.o build_HF_mat.o gen_Lebedev_grid.o gen_int_grid.o \
        eval_XC_func.o build_XCmat.o CDIIS.o build_Dmat.o 

$(LIB): $(OBJS) 
	$(AR) $@ $^

CMS_basis.o: Makefile CMS_basis.h CMS_basis.c 
	$(CC) $(CFLAGS) $(INCS) -c CMS_basis.c -o $@ 

CMS_simint.o: Makefile CMS_simint.h CMS_simint.c 
	$(CC) $(CFLAGS) $(INCS) -c CMS_simint.c -o $@ 

utils.o: Makefile utils.h utils.c 
	$(CC) $(CFLAGS) $(INCS) -c utils.c -o $@ 

ket_sp_list.o: Makefile ket_sp_list.h ket_sp_list.c 
	$(CC) $(CFLAGS) $(INCS) -c ket_sp_list.c -o $@ 

TinyDFT_typedef.o: Makefile utils.h TinyDFT_typedef.h TinyDFT_typedef.c 
	$(CC) $(CFLAGS) $(INCS) -c TinyDFT_typedef.c -o $@ 

acc_JKmat.o: Makefile TinyDFT_typedef.h acc_JKmat.h acc_JKmat.c 
	$(CC) $(CFLAGS) $(INCS) -c acc_JKmat.c -o $@ 

build_HF_mat.o: Makefile TinyDFT_typedef.h ket_sp_list.h build_HF_mat.h build_HF_mat.c 
	$(CC) $(CFLAGS) $(INCS) -c build_HF_mat.c -o $@ 

gen_Lebedev_grid.o: Makefile gen_Lebedev_grid.h gen_Lebedev_grid.c 
	$(CC) $(CFLAGS) $(INCS) -c gen_Lebedev_grid.c -o $@ 

gen_int_grid.o: Makefile gen_Lebedev_grid.h gen_int_grid.h gen_int_grid.c 
	$(CC) $(CFLAGS) $(INCS) -c gen_int_grid.c -o $@ 

eval_XC_func.o: Makefile gen_int_grid.h eval_XC_func.h eval_XC_func.c 
	$(CC) $(CFLAGS) $(INCS) -c eval_XC_func.c -o $@ 

build_XCmat.o: Makefile gen_int_grid.h eval_XC_func.h build_XCmat.h build_XCmat.c 
	$(CC) $(CFLAGS) $(INCS) -c build_XCmat.c -o $@ 

CDIIS.o: Makefile TinyDFT_typedef.h CDIIS.h CDIIS.c 
	$(CC) $(CFLAGS) $(INCS) -c CDIIS.c -o $@ 

build_Dmat.o: Makefile TinyDFT_typedef.h build_Dmat.h build_Dmat.c 
	$(CC) $(CFLAGS) $(INCS) -c build_Dmat.c -o $@ 

clean:
	rm -f *.o $(LIB)
