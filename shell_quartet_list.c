#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "shell_quartet_list.h"

// Sanity check for parameters is skipped for better performance

void init_KetShellpairList_with_buffer(KetShellpairList_t ket_shellpair_list, int *PQlist_buffer)
{
    ket_shellpair_list->npairs = 0;
    ket_shellpair_list->P_list = PQlist_buffer;
    ket_shellpair_list->Q_list = PQlist_buffer + MAX_LIST_SIZE;
}

void create_ThreadKetShellpairLists(ThreadKetShellpairLists_t *_thread_ket_sp_lists)
{
    ThreadKetShellpairLists_t thread_ket_sp_lists;
    thread_ket_sp_lists = (ThreadKetShellpairLists_t) malloc(sizeof(ThreadKetShellpairLists));
    
    void *ptr = malloc(sizeof(KetShellpairList) * MAX_AM_PAIRS);
    thread_ket_sp_lists->ket_shellpair_lists = (KetShellpairList_t) ptr;
    assert(thread_ket_sp_lists->ket_shellpair_lists != NULL);
    
    thread_ket_sp_lists->bufptr = (int*) malloc(sizeof(int) * MAX_AM_PAIRS * MAX_LIST_SIZE * 2);
    assert(thread_ket_sp_lists->bufptr != NULL);
    
    for (int i = 0; i < MAX_AM_PAIRS; i++)
    {
        init_KetShellpairList_with_buffer(
            &thread_ket_sp_lists->ket_shellpair_lists[i],
            thread_ket_sp_lists->bufptr + i * MAX_LIST_SIZE * 2
        );
    }
    
    *_thread_ket_sp_lists = thread_ket_sp_lists;
}

void free_ThreadKetShellpairLists(ThreadKetShellpairLists_t thread_ket_sp_lists)
{
    free(thread_ket_sp_lists->ket_shellpair_lists);
    free(thread_ket_sp_lists->bufptr);
    free(thread_ket_sp_lists);
}

