#ifndef _YATSCF_KET_SP_LIST_H_
#define _YATSCF_KET_SP_LIST_H_

#define MAX_LIST_SIZE 16   // == SIMINT_NSHELL_SIMD 
#define MAX_AM_PAIRS  25   // == SLEN * SLEN in cint_basisset.c

typedef struct
{
    // Number of shell pairs in the list
    int npairs;
    
    // (P_list[i], Q_list[i]) are the shell pair ids for ket-side
    // AM(P_list[]) are the same, AM(Q_list[]) are the same
    int *P_list, *Q_list;
} KetShellpairList;

typedef KetShellpairList* KetShellpairList_t;

typedef struct 
{
    // Shell pair lists for different AM pairs
    KetShellpairList *ket_shellpair_lists;  
    
    // Pointer to the memory space for all KetShellpairLists' storage
    int *bufptr;
} ThreadKetShellpairLists;

typedef ThreadKetShellpairLists* ThreadKetShellpairLists_t;

// Initialize a KetShellpairList with a given buffer for storing P_list and Q_list
void init_KetShellpairList_with_buffer(KetShellpairList_t ket_shellpair_list, int *PQlist_buffer);

// Add a ket-side shell pair to a KetShellpairList
// Frequently used, inline it
static inline void add_shellpair_to_KetShellPairList(KetShellpairList_t ket_shellpair_list, int P, int Q)
{
    int index = ket_shellpair_list->npairs;
    ket_shellpair_list->P_list[index] = P;
    ket_shellpair_list->Q_list[index] = Q;
    ket_shellpair_list->npairs++;
}

// We don't need a reset function for KetShellpairList, since we just need to set npairs = 0

// Create a ThreadKetShellpairLists and initialize all its KetShellpairLists
void create_ThreadKetShellpairLists(ThreadKetShellpairLists_t *_thread_ket_sp_lists);

// Free a ThreadKetShellpairLists and all its KetShellpairLists, release the memory
void free_ThreadKetShellpairLists(ThreadKetShellpairLists_t thread_ket_sp_lists);

// We don't need a reset function for ThreadKetShellpairLists, since all KetShellpairLists
// will be checked and reset after each MN iteration

#endif
