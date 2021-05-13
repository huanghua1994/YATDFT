CC           = gcc
USE_MKL      = 0
USE_OPENBLAS = 1

include common.make
USE_AARCH64_SVE = 0
SVE_VECTOR_BITS = 512
ifeq ($(strip $(USE_AARCH64_SVE)), 1)
CFLAGS := $(subst -march=native, -march=armv8.2-a+sve -msve-vector-bits=$(SVE_VECTOR_BITS), $(CFLAGS))
endif
