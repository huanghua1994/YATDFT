#ifndef __CMS_CONFIG_H__
#define __CMS_CONFIG_H__

#include <stdio.h>
#include <unistd.h>

#ifndef __APPLE__
#define HAS_MALLOC_H
#endif

#define _DEBUG_LEVEL_    1   // 0 to 10, 0 is no debug print info at all, 10 is full info

static inline void *CMS_malloc_aligned(size_t size, size_t alignment)
{
    void *ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
}

static inline void CMS_free_aligned(void *mem)
{
    free(mem);
}

#define ALIGNED_MALLOC(size)  CMS_malloc_aligned(size, __ALIGNLEN__)
#define ALIGNED_FREE(addr)    CMS_free_aligned(addr)

#if ( _DEBUG_LEVEL_ == -1 )
#define CMS_PRINTF( level, fmt, args... )        {}
#else
#define CMS_PRINTF( level, fmt, args... )                                           \
        do                                                                          \
        {                                                                           \
            if ( (unsigned)(level) <= _DEBUG_LEVEL_ )                               \
            {                                                                       \
                sprintf( basis->str_buf, "%s() line %d ", __FUNCTION__, __LINE__ ); \
                sprintf( basis->str_buf + strlen(basis->str_buf), fmt, ##args );    \
                fprintf( stdout, "%s", basis->str_buf );                            \
                fflush( stdout );                                                   \
            }                                                                       \
        } while ( 0 )
#endif


#if ( _DEBUG_LEVEL_ > 1 )
#define CMS_INFO( fmt, args... )                                             \
        do                                                                   \
        {                                                                    \
            sprintf( basis->str_buf, "**** CMS: ");                          \
            sprintf( basis->str_buf + strlen("**** CMS: "), fmt, ##args );   \
            fprintf( stdout, "%s", basis->str_buf );                         \
            fflush( stdout );                                                \
        } while ( 0 )
#else
#define CMS_INFO( fmt, args... )        {}
#endif

#define CMS_ASSERT(condition) if (!(condition)) { \
    dprintf(2, "ASSERTION FAILED: %s in %s:%d\n", #condition, __FILE__, __LINE__); \
    fsync(2); \
    abort(); \
}

typedef enum
{
    CMS_STATUS_SUCCESS          = 0,
    CMS_STATUS_NOT_INITIALIZED  = 1,
    CMS_STATUS_ALLOC_FAILED     = 2,
    CMS_STATUS_INVALID_VALUE    = 3,
    CMS_STATUS_EXECUTION_FAILED = 4,
    CMS_STATUS_INTERNAL_ERROR   = 5,
    CMS_STATUS_FILEIO_FAILED    = 6,
    CMS_STATUS_OFFLOAD_ERROR    = 7
} CMSStatus_t;

#endif
