CC = icc

USE_MKL      = 1
USE_OPENBLAS = 0
USE_LIBXC    = 0

YATDFT_INSTALL_DIR   = ..
LIBXC_INSTALL_DIR    = /home/mkurisu/Workspace/libxc/install
SIMINT_INSTALL_DIR   = /home/mkurisu/Workspace/simint/build-avx/install
OPENBLAS_INSTALL_DIR = /home/mkurisu/Workspace/OpenBLAS-0.3.8/install

INCS     = -I$(YATDFT_INSTALL_DIR)/src/include -I$(SIMINT_INSTALL_DIR)/include 
LIBS     = $(YATDFT_INSTALL_DIR)/src/libYATDFT.a $(SIMINT_INSTALL_DIR)/lib64/libsimint.a
CFLAGS   = $(INCS) -Wall -g -std=gnu99 -O3 
LDFLAGS  = -g -O3

ifeq ($(CC), icc)
CFLAGS  += -qopenmp -xHost
LDFLAGS += -qopenmp -xHost
endif

ifeq ($(CC), gcc)
CFLAGS  += -fopenmp -march=native -Wno-unused-result -Wno-unused-function
LDFLAGS += -fopenmp -march=native -lm
endif

ifeq ($(USE_MKL), 1)
CFLAGS  += -DUSE_MKL -mkl
LDFLAGS += -mkl
endif

ifeq ($(USE_OPENBLAS), 1)
CFLAGS  += -DUSE_OPENBLAS
LIBS    += $(OPENBLAS_INSTALL_DIR)/lib/libopenblas.a
INCS    += -I$(OPENBLAS_INSTALL_DIR)/include
LDFLAGS += -lm
endif

ifeq ($(USE_LIBXC), 1)
INCS    += -I$(LIBXC_INSTALL_DIR)/include
CFLAGS  += -DUSE_LIBXC
LDFLAGS += -L$(LIBXC_INSTALL_DIR)/lib -lxc
endif

SRCS = AnySCF.c
OBJS = $(SRCS:.c=.o)
EXES = $(OBJS:.o=.exe)

all: $(EXES)

%.o: %.c
	$(CC) $(CFLAGS)  -c $^ 

%.exe: %.o $(LIBS)
	$(CC) -o $@ $^ $(LDFLAGS) 

clean:
	rm -f $(EXES) $(OBJS)
