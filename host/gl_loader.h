#pragma once
#include <GL/glcorearb.h>   // types + enums + PFN typedefs, no function definitions

// Every modern GL function we call, loaded by name at runtime.
#define TERNOISE_GL_FUNCS(X) \
    X(PFNGLCREATESHADERPROC,            CreateShader) \
    X(PFNGLSHADERSOURCEPROC,            ShaderSource) \
    X(PFNGLCOMPILESHADERPROC,           CompileShader) \
    X(PFNGLGETSHADERIVPROC,             GetShaderiv) \
    X(PFNGLGETSHADERINFOLOGPROC,        GetShaderInfoLog) \
    X(PFNGLCREATEPROGRAMPROC,           CreateProgram) \
    X(PFNGLATTACHSHADERPROC,            AttachShader) \
    X(PFNGLLINKPROGRAMPROC,             LinkProgram) \
    X(PFNGLGETPROGRAMIVPROC,            GetProgramiv) \
    X(PFNGLGETPROGRAMINFOLOGPROC,       GetProgramInfoLog) \
    X(PFNGLDELETESHADERPROC,            DeleteShader) \
    X(PFNGLUSEPROGRAMPROC,              UseProgram) \
    X(PFNGLGETUNIFORMLOCATIONPROC,      GetUniformLocation) \
    X(PFNGLUNIFORM1IPROC,               Uniform1i) \
    X(PFNGLUNIFORM1UIPROC,              Uniform1ui) \
    X(PFNGLUNIFORM1FPROC,               Uniform1f) \
    X(PFNGLUNIFORM3FPROC,               Uniform3f) \
    X(PFNGLGENTEXTURESPROC,             GenTextures) \
    X(PFNGLBINDTEXTUREPROC,             BindTexture) \
    X(PFNGLTEXSTORAGE2DPROC,            TexStorage2D) \
    X(PFNGLTEXPARAMETERIPROC,           TexParameteri) \
    X(PFNGLACTIVETEXTUREPROC,           ActiveTexture) \
    X(PFNGLBINDIMAGETEXTUREPROC,        BindImageTexture) \
    X(PFNGLDISPATCHCOMPUTEPROC,         DispatchCompute) \
    X(PFNGLMEMORYBARRIERPROC,           MemoryBarrier) \
    X(PFNGLGENVERTEXARRAYSPROC,         GenVertexArrays) \
    X(PFNGLBINDVERTEXARRAYPROC,         BindVertexArray) \
    X(PFNGLDRAWARRAYSPROC,              DrawArrays) \
    X(PFNGLGETTEXIMAGEPROC,             GetTexImage) \
    X(PFNGLVIEWPORTPROC,                Viewport) \
    X(PFNGLCLEARCOLORPROC,              ClearColor) \
    X(PFNGLCLEARPROC,                   Clear) \
    X(PFNGLGETSTRINGPROC,               GetString) \
    X(PFNGLFINISHPROC,                  Finish)

namespace gl {
#define TERNOISE_DECL(type, name) extern type name;
TERNOISE_GL_FUNCS(TERNOISE_DECL)
#undef TERNOISE_DECL
// Call after glfwMakeContextCurrent. Aborts naming the symbol if a lookup fails.
void load(void* (*get_proc)(const char*));
}
