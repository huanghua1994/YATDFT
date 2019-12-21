alignlen = 64

CC  = icc
AR  = xiar rcs

SIMINT_DIR = /home/mkurisu/Workspace/simint/build-avx/install

INC=-I. -I$(SIMINT_DIR)/include

SRC     = CMS_BasisSet.c CMS_Simint.c
CFLAGS  = -O3 -xHost -Wall -qopenmp -g -std=gnu99 -m64
CFLAGS += -D__ALIGNLEN__=${alignlen}

LIBCMS = libCMS.a
OBJS := $(addsuffix .o, $(basename $(SRC)))

all: $(LIBCMS)

$(LIBCMS): $(OBJS)
	$(AR) $@ $^

%.o : %.c Makefile
	$(CC) $(CFLAGS) $(INC) -c $< -o $@ 

clean:
	rm -f *.o *.s *.d *~ *.a
