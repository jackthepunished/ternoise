#include "shader_util.h"
#include "gl_loader.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

static std::string slurp(const char* path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot open shader %s\n", path); std::exit(2); }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static GLuint compile(GLenum type, const char* path) {
    const std::string src = slurp(path);
    const char* p = src.c_str();
    GLuint s = gl::CreateShader(type);
    gl::ShaderSource(s, 1, &p, nullptr);
    gl::CompileShader(s);
    GLint ok = 0;
    gl::GetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        gl::GetShaderInfoLog(s, sizeof log, nullptr, log);
        std::fprintf(stderr, "compile error in %s:\n%s\n", path, log);
        std::exit(2);
    }
    return s;
}

static GLuint link(GLuint a, GLuint b) {
    GLuint prog = gl::CreateProgram();
    gl::AttachShader(prog, a);
    if (b) gl::AttachShader(prog, b);
    gl::LinkProgram(prog);
    GLint ok = 0;
    gl::GetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        gl::GetProgramInfoLog(prog, sizeof log, nullptr, log);
        std::fprintf(stderr, "link error:\n%s\n", log);
        std::exit(2);
    }
    gl::DeleteShader(a);
    if (b) gl::DeleteShader(b);
    return prog;
}

GLuint make_program_graphics(const char* vs_path, const char* fs_path) {
    return link(compile(GL_VERTEX_SHADER, vs_path), compile(GL_FRAGMENT_SHADER, fs_path));
}

GLuint make_program_compute(const char* cs_path) {
    return link(compile(GL_COMPUTE_SHADER, cs_path), 0);
}
