// ternoise host: noisy 1-spp path tracer + G-buffers + dataset dump.
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "gl_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    int size = 256, frames = 0;
    bool hidden = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--size") && i + 1 < argc) size = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) frames = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--hidden")) hidden = true;
        else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (size <= 0 || size % 8) { std::fprintf(stderr, "--size must be a positive multiple of 8\n"); return 2; }

    if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 2; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    if (hidden) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* win = glfwCreateWindow(size, size, "ternoise host", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "GL 4.3 window/context creation failed\n"); glfwTerminate(); return 2; }
    glfwMakeContextCurrent(win);
    gl::load([](const char* n) -> void* { return (void*)glfwGetProcAddress(n); });
    std::printf("GL_VERSION : %s\n", (const char*)gl::GetString(GL_VERSION));
    std::printf("GL_RENDERER: %s\n", (const char*)gl::GetString(GL_RENDERER));

    int rendered = 0;
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
        gl::Viewport(0, 0, size, size);
        gl::ClearColor(0.10f, 0.10f, 0.12f, 1.0f);
        gl::Clear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(win);
        if (frames && ++rendered >= frames) break;
    }
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
