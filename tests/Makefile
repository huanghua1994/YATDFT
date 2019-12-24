CC = icc

BLAS_LIB       = -mkl=parallel
SIMINT_INCDIR  = /home/mkurisu/Workspace/simint/build-avx/install/include
SIMINT_LIBFILE = /home/mkurisu/Workspace/simint/build-avx/install/lib/libsimint.a

LIBS     = ../src/libYATDFT.a $(SIMINT_LIBFILE)
INCS     = -I../src/ -I$(SIMINT_INCDIR)
CFLAGS   = $(INCS) -Wall -g -std=gnu99 -O3 -qopenmp -xHost -mkl $(BLAS_LIB)
LDFLAGS  = -g -O3 -xHost -qopenmp -mkl $(BLAS_LIB)

SRCS = HFSCF.c KSDFT.c
OBJS = $(SRCS:.c=.o)
EXES = $(OBJS:.o=.x)

all: $(EXES)

%.o: %.c
	$(CC) $(CFLAGS)  -c $^ 

%.x: %.o $(LIBS)
	$(CC) $(LDFLAGS) -o $@ $^ 

clean:
	rm -f $(EXES) $(OBJS)
