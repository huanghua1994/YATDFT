LIBXC_INSTALL_DIR    = /home/mkurisu/Workspace/libxc/install
SIMINT_INSTALL_DIR   = /home/mkurisu/Workspace/simint/build-avx/install
YATDFT_INSTALL_DIR   = /home/mkurisu/Workspace/YATDFT
OPENBLAS_INSTALL_DIR = /home/mkurisu/Workspace/OpenBLAS-git/install

DEFS    = 
INCS    = -I$(YATDFT_INSTALL_DIR)/include -I$(SIMINT_INSTALL_DIR)/include
CFLAGS  = $(INCS) -Wall -g -std=gnu11 -O3 -fPIC $(DEFS)
LDFLAGS = -g -O3 -fopenmp
LIBS    = $(YATDFT_INSTALL_DIR)/lib/libYATDFT.a $(SIMINT_INSTALL_DIR)/lib64/libsimint.a 

ifeq ($(shell $(CC) --version 2>&1 | grep -c "icc"), 1)
CFLAGS  += -fopenmp -xHost
endif

ifeq ($(shell $(CC) --version 2>&1 | grep -c "gcc"), 1)
CFLAGS  += -fopenmp -march=native -Wno-unused-result -Wno-unused-function
LIBS    += -lgfortran -lm
endif

ifeq ($(strip $(USE_MKL)), 1)
DEFS    += -DUSE_MKL
CFLAGS  += -mkl
LDFLAGS += -mkl
endif

ifeq ($(strip $(USE_OPENBLAS)), 1)
DEFS    += -DUSE_OPENBLAS
INCS    += -I$(OPENBLAS_INSTALL_DIR)/include
LDFLAGS += -L$(OPENBLAS_INSTALL_DIR)/lib
LIBS    += -lopenblas
endif

ifeq ($(USE_LIBXC), 1)
CFLAGS  += -DUSE_LIBXC
INCS    += -I$(LIBXC_INSTALL_DIR)/include
LDFLAGS += -L$(LIBXC_INSTALL_DIR)/lib -lxc
endif

C_SRCS 	= $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.c.o)
EXES    = $(C_SRCS:.c=.exe)

# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(C_OBJS)

all: $(EXES)

%.c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

%.exe: %.c.o $(LIB)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(EXES) $(C_OBJS)
