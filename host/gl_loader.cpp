#include "gl_loader.h"
#include <cstdio>
#include <cstdlib>

namespace gl {
#define TERNOISE_DEF(type, name) type name = nullptr;
TERNOISE_GL_FUNCS(TERNOISE_DEF)
#undef TERNOISE_DEF

void load(void* (*get_proc)(const char*)) {
#define TERNOISE_LOAD(type, name) \
    name = (type)get_proc("gl" #name); \
    if (!name) { std::fprintf(stderr, "missing GL symbol gl" #name "\n"); std::exit(2); }
    TERNOISE_GL_FUNCS(TERNOISE_LOAD)
#undef TERNOISE_LOAD
}
}  // namespace gl
