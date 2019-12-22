CC  = icc
AR  = xiar rcs
LIB = libYATDFT.a

BLAS_LIB       = -mkl=parallel
SIMINT_INCDIR  = /home/mkurisu/Workspace/simint/build-avx/install/include
SIMINT_LIBFILE = /home/mkurisu/Workspace/simint/build-avx/install/lib/libsimint.a

DEFS    = -D__ALIGNLEN__=64
INCS    = -I./ -I$(SIMINT_INCDIR)
LIBS    = $(BLAS_LIB) $(SIMINT_LIBFILE) 

CFLAGS  = -Wall -g -O3 -qopenmp -std=gnu99 -xHost $(DEFS) $(BLAS_LIB)
LDFLAGS = -lpthread -qopenmp $(BLAS_LIB)

OBJS =  CMS_basis.o CMS_simint.o utils.o build_Dmat.o ket_sp_list.o \
		acc_JKmat.o build_HF_mat.o CDIIS.o TinyDFT_typedef.o

$(LIB): $(OBJS) 
	$(AR) $@ $^

CMS_basis.o: Makefile CMS_basis.c CMS_basis.h
	$(CC) $(CFLAGS) $(INCS) -c CMS_basis.c -o $@ 

CMS_simint.o: Makefile CMS_simint.c CMS_simint.h
	$(CC) $(CFLAGS) $(INCS) -c CMS_simint.c -o $@ 

utils.o: Makefile utils.c utils.h
	$(CC) $(CFLAGS) $(INCS) -c utils.c -o $@ 

build_Dmat.o: Makefile build_Dmat.c build_Dmat.h TinyDFT_typedef.h
	$(CC) $(CFLAGS) $(INCS) -c build_Dmat.c -o $@ 

ket_sp_list.o: Makefile ket_sp_list.c ket_sp_list.h
	$(CC) $(CFLAGS) $(INCS) -c ket_sp_list.c -o $@ 

acc_JKmat.o: Makefile acc_JKmat.h acc_JKmat.c TinyDFT_typedef.h
	$(CC) $(CFLAGS) $(INCS) -c acc_JKmat.c -o $@ 

build_HF_mat.o: Makefile build_HF_mat.c build_HF_mat.h TinyDFT_typedef.h ket_sp_list.h
	$(CC) $(CFLAGS) $(INCS) -c build_HF_mat.c -o $@ 

CDIIS.o: Makefile CDIIS.c CDIIS.h TinyDFT_typedef.h
	$(CC) $(CFLAGS) $(INCS) -c CDIIS.c -o $@ 

TinyDFT_typedef.o: Makefile TinyDFT_typedef.c TinyDFT_typedef.h utils.h
	$(CC) $(CFLAGS) $(INCS) -c TinyDFT_typedef.c -o $@ 

clean:
	rm -f *.o $(LIB)
