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

OBJS = utils.o build_density.o shell_quartet_list.o Accum_Fock.o build_Fock.o DIIS.o TinySCF.o main.o 

$(EXE): Makefile $(OBJS) $(LIBCMS_LIBFILE) $(SIMINT_LIBFILE)
	$(CC) $(CFLAGS) ${LDFLAGS} $(OBJS) -o $(EXE) $(LIBS)

utils.o: Makefile utils.c utils.h
	$(CC) $(CFLAGS) $(INCS) -c utils.c -o $@ 

build_density.o: Makefile build_density.c build_density.h TinySCF.h
	$(CC) $(CFLAGS) $(INCS) $(BLAS_LIB) -c build_density.c -o $@ 

shell_quartet_list.o: Makefile shell_quartet_list.c shell_quartet_list.h
	$(CC) $(CFLAGS) $(INCS) -c shell_quartet_list.c -o $@ 

Accum_Fock.o: Makefile Accum_Fock.h Accum_Fock.c TinySCF.h
	$(CC) $(CFLAGS) $(INCS) -c Accum_Fock.c -o $@ 

build_Fock.o: Makefile build_Fock.c build_Fock.h TinySCF.h shell_quartet_list.h
	$(CC) $(CFLAGS) $(INCS) -c build_Fock.c -o $@ 

DIIS.o: Makefile DIIS.c DIIS.h TinySCF.h
	$(CC) $(CFLAGS) $(INCS) -c DIIS.c -o $@ 

TinySCF.o: Makefile TinySCF.c TinySCF.h utils.h
	$(CC) $(CFLAGS) $(INCS) $(BLAS_LIB) -c TinySCF.c -o $@ 
	
main.o: Makefile main.c TinySCF.h
	$(CC) $(CFLAGS) $(INCS) -c main.c    -o $@ 

clean:
	rm -f *.o $(EXE)
