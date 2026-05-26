#include "context.h"
#include <cstring>

static uint64_t* aligned_alloc_words(size_t n_words)
{
    void* p = nullptr;
    if (posix_memalign(&p, 64, n_words * sizeof(uint64_t)) != 0) return nullptr;
    std::memset(p, 0, n_words * sizeof(uint64_t));
    return reinterpret_cast<uint64_t*>(p);
}

void KernelContext::alloc(size_t tw)
{
    tile_words = tw;
    rs_store  = aligned_alloc_words(5 * 3 * tw);
    C_store   = aligned_alloc_words(5 * tw);
    adult_tmp = aligned_alloc_words(tw);
}

void KernelContext::free_data()
{
    std::free(rs_store);  rs_store  = nullptr;
    std::free(C_store);   C_store   = nullptr;
    std::free(adult_tmp); adult_tmp = nullptr;
    tile_words = 0;
}
