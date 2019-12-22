CC  = icc
EXE = TinySCF.exe

BLAS_LIB       = -mkl=parallel
SIMINT_INCDIR  = /home/mkurisu/Workspace/simint/build-avx/install/include
SIMINT_LIBFILE = /home/mkurisu/Workspace/simint/build-avx/install/lib/libsimint.a
LIBCMS_INCDIR  = ./libCMS
LIBCMS_LIBFILE = ./libCMS/libCMS.a

DEFS    = -DBUILD_J_MAT_STD -DBUILD_K_MAT_HF
INCS    = -I./ -I$(SIMINT_INCDIR) -I$(LIBCMS_INCDIR)
LIBS    = $(BLAS_LIB) $(LIBCMS_LIBFILE) $(SIMINT_LIBFILE) 

CFLAGS  = -Wall -g -O3 -qopenmp -std=gnu99 -xHost $(DEFS)
LDFLAGS = -L$(LIBCMS_LIBFILE) -lpthread -qopenmp

OBJS = utils.o build_Dmat.o ket_sp_list.o acc_JKmat.o build_HF_mat.o DIIS.o TinySCF_typedef.o main.o 

$(EXE): Makefile $(OBJS) $(LIBCMS_LIBFILE) $(SIMINT_LIBFILE)
	$(CC) $(CFLAGS) ${LDFLAGS} $(OBJS) -o $(EXE) $(LIBS)

utils.o: Makefile utils.c utils.h
	$(CC) $(CFLAGS) $(INCS) -c utils.c -o $@ 

build_Dmat.o: Makefile build_Dmat.c build_Dmat.h TinySCF_typedef.h
	$(CC) $(CFLAGS) $(INCS) $(BLAS_LIB) -c build_Dmat.c -o $@ 

ket_sp_list.o: Makefile ket_sp_list.c ket_sp_list.h
	$(CC) $(CFLAGS) $(INCS) -c ket_sp_list.c -o $@ 

acc_JKmat.o: Makefile acc_JKmat.h acc_JKmat.c TinySCF_typedef.h
	$(CC) $(CFLAGS) $(INCS) -c acc_JKmat.c -o $@ 

build_HF_mat.o: Makefile build_HF_mat.c build_HF_mat.h TinySCF_typedef.h ket_sp_list.h
	$(CC) $(CFLAGS) $(INCS) -c build_HF_mat.c -o $@ 

DIIS.o: Makefile DIIS.c DIIS.h TinySCF_typedef.h
	$(CC) $(CFLAGS) $(INCS) -c DIIS.c -o $@ 

TinySCF_typedef.o: Makefile TinySCF_typedef.c TinySCF_typedef.h utils.h
	$(CC) $(CFLAGS) $(INCS) $(BLAS_LIB) -c TinySCF_typedef.c -o $@ 
	
main.o: Makefile main.c TinySCF_typedef.h build_Dmat.h build_HF_mat.h
	$(CC) $(CFLAGS) $(INCS) -c main.c    -o $@ 

clean:
	rm -f *.o $(EXE)
