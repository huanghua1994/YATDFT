LIB_A   = libYATDFT.a
LIB_SO  = libYATDFT.so

C_SRCS  = $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.c.o)

SIMINT_INSTALL_DIR   = /home/mkurisu/Workspace/simint/build-avx/install
LIBXC_INSTALL_DIR    = /home/mkurisu/Workspace/libxc/install
OPENBLAS_INSTALL_DIR = /home/mkurisu/Workspace/OpenBLAS-git/install

DEFS    = -D__ALIGNLEN__=64
INCS    = -I$(SIMINT_INSTALL_DIR)/include
CFLAGS  = $(INCS) -Wall -g -std=gnu11 -O3 -fPIC $(DEFS)

ifeq ($(shell $(CC) --version 2>&1 | grep -c "icc"), 1)
AR      = xiar rcs
CFLAGS += -qopenmp -xHost
endif

ifeq ($(shell $(CC) --version 2>&1 | grep -c "gcc"), 1)
AR      = ar rcs
CFLAGS += -fopenmp -march=native -Wno-unused-result -Wno-unused-function
endif

ifeq ($(strip $(USE_MKL)), 1)
DEFS   += -DUSE_MKL
CFLAGS += -mkl
endif

ifeq ($(strip $(USE_OPENBLAS)), 1)
DEFS   += -DUSE_OPENBLAS
INCS   += -I$(OPENBLAS_INSTALL_DIR)/include
endif

ifeq ($(USE_LIBXC), 1)
INCS   += -I$(LIBXC_INSTALL_DIR)/include
DEFS   += -DUSE_LIBXC
endif

# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(C_OBJS)

all: install

install: $(LIB_A) $(LIB_SO)
	mkdir -p ../lib
	mkdir -p ../include
	cp -u $(LIB_A)  ../lib/$(LIB_A)
	cp -u $(LIB_SO) ../lib/$(LIB_SO)
	cp -u *.h ../include/

$(LIB_A): $(C_OBJS) 
	$(AR) $@ $^

$(LIB_SO): $(C_OBJS) 
	$(CC) -shared -o $@ $^

%.c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	rm $(C_OBJS) $(LIB_A) $(LIB_SO)
