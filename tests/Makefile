CC = icc

USE_LIBXC = 1

BLAS_LIB       = -mkl=parallel
SIMINT_INCDIR  = /home/mkurisu/Workspace/simint/build-avx/install/include
SIMINT_LIBFILE = /home/mkurisu/Workspace/simint/build-avx/install/lib/libsimint.a
LIBXC_INCDIR   = /home/mkurisu/Workspace/libxc/install/include
LIBXC_LIBDIR   = /home/mkurisu/Workspace/libxc/install/lib

LIBS     = ../src/libYATDFT.a $(SIMINT_LIBFILE)
INCS     = -I../src/ -I$(SIMINT_INCDIR)
CFLAGS   = $(INCS) -Wall -g -std=gnu99 -O3 -qopenmp -xHost $(BLAS_LIB)
LDFLAGS  = -g -O3 -xHost -qopenmp $(BLAS_LIB)

ifeq ($(USE_LIBXC), 1)
INCS    += -I$(LIBXC_INCDIR)
CFLAGS  += -DUSE_LIBXC
LDFLAGS += -L$(LIBXC_LIBDIR) -lxc
endif

SRCS = HFSCF.c KSDFT.c HFSCF-DF.c
OBJS = $(SRCS:.c=.o)
EXES = $(OBJS:.o=.exe)

all: $(EXES)

%.o: %.c
	$(CC) $(CFLAGS)  -c $^ 

%.exe: %.o $(LIBS)
	$(CC) $(LDFLAGS) -o $@ $^ 

clean:
	rm -f $(EXES) $(OBJS)
