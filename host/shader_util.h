#pragma once
#include <GL/glcorearb.h>

// Compile+link helpers. On any error: print the info log, exit 2.
GLuint make_program_graphics(const char* vs_path, const char* fs_path);
GLuint make_program_compute(const char* cs_path);
