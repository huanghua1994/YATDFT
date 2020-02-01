CC = icc

USE_LIBXC  = 0

BLAS_LIB   = -mkl=parallel
LIBXC_DIR  = /home/mkurisu/Workspace/libxc/install
YATDFT_DIR = ..
SIMINT_DIR = /home/mkurisu/Workspace/simint/build-avx/install


LIBS     = $(YATDFT_DIR)/src/libYATDFT.a $(SIMINT_DIR)/lib64/libsimint.a
INCS     = -I$(YATDFT_DIR)/src/include -I$(SIMINT_DIR)/include 
CFLAGS   = $(INCS) -Wall -g -std=gnu99 -O3 -qopenmp -xHost $(BLAS_LIB)
LDFLAGS  = -g -O3 -xHost -qopenmp $(BLAS_LIB)

ifeq ($(USE_LIBXC), 1)
INCS    += -I$(LIBXC_DIR)/include
CFLAGS  += -DUSE_LIBXC
LDFLAGS += -L$(LIBXC_DIR)/lib -lxc
endif

SRCS = AnySCF.c
OBJS = $(SRCS:.c=.o)
EXES = $(OBJS:.o=.exe)

all: $(EXES)

%.o: %.c
	$(CC) $(CFLAGS)  -c $^ 

%.exe: %.o $(LIBS)
	$(CC) $(LDFLAGS) -o $@ $^ 

clean:
	rm -f $(EXES) $(OBJS)
