// ternoise host: noisy 1-spp path tracer + G-buffers + dataset dump.
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "gl_loader.h"
#include "shader_util.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static GLuint make_tex(GLenum ifmt, int size) {
    GLuint t;
    gl::GenTextures(1, &t);
    gl::BindTexture(GL_TEXTURE_2D, t);
    gl::TexStorage2D(GL_TEXTURE_2D, 1, ifmt, size, size);
    gl::TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    gl::TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return t;
}

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

    // image unit conventions: 0 = accum RGBA32F, 1 = normal RGBA32F, 2 = depth R32F
    GLuint tex_accum = make_tex(GL_RGBA32F, size);
    GLuint tex_normal = make_tex(GL_RGBA32F, size);
    GLuint tex_depth = make_tex(GL_R32F, size);
    gl::BindImageTexture(0, tex_accum, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    gl::BindImageTexture(1, tex_normal, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    gl::BindImageTexture(2, tex_depth, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    GLuint prog_cs = make_program_compute("host/shaders/pathtrace.comp");
    GLuint prog_blit = make_program_graphics("host/shaders/blit.vert", "host/shaders/blit.frag");
    GLint u_tex = gl::GetUniformLocation(prog_blit, "tex");
    GLint u_mode = gl::GetUniformLocation(prog_blit, "display_mode");
    GLint u_inv_spp = gl::GetUniformLocation(prog_blit, "inv_spp");
    GLint u_frame = gl::GetUniformLocation(prog_cs, "frame_index");
    GLint u_vseed = gl::GetUniformLocation(prog_cs, "view_seed");
    GLint u_cpos = gl::GetUniformLocation(prog_cs, "cam_pos");
    GLint u_ctgt = gl::GetUniformLocation(prog_cs, "cam_target");
    GLuint vao;
    gl::GenVertexArrays(1, &vao);

    const float cam_pos[3] = {0.0f, 2.0f, 6.5f};
    const float cam_target[3] = {0.0f, 1.6f, 0.0f};
    unsigned frame_index = 0;
    int view = 0;   // 0 accum, 1 normal, 2 depth
    int rendered = 0;
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
        if (glfwGetKey(win, GLFW_KEY_1) == GLFW_PRESS) view = 0;
        if (glfwGetKey(win, GLFW_KEY_2) == GLFW_PRESS) view = 1;
        if (glfwGetKey(win, GLFW_KEY_3) == GLFW_PRESS) view = 2;
        if (glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS) frame_index = 0;

        gl::UseProgram(prog_cs);
        gl::Uniform1ui(u_frame, frame_index);
        gl::Uniform1ui(u_vseed, 1u);
        gl::Uniform3f(u_cpos, cam_pos[0], cam_pos[1], cam_pos[2]);
        gl::Uniform3f(u_ctgt, cam_target[0], cam_target[1], cam_target[2]);
        gl::DispatchCompute((GLuint)size / 8, (GLuint)size / 8, 1);
        gl::MemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        ++frame_index;

        gl::Viewport(0, 0, size, size);
        gl::UseProgram(prog_blit);
        gl::ActiveTexture(GL_TEXTURE0);
        gl::BindTexture(GL_TEXTURE_2D, view == 0 ? tex_accum : view == 1 ? tex_normal : tex_depth);
        gl::Uniform1i(u_tex, 0);
        gl::Uniform1i(u_mode, view == 0 ? 0 : 1);
        gl::Uniform1f(u_inv_spp, view == 0 ? 1.0f / (float)frame_index : 1.0f);
        gl::BindVertexArray(vao);
        gl::DrawArrays(GL_TRIANGLES, 0, 3);
        glfwSwapBuffers(win);
        if (frames && ++rendered >= frames) break;
    }
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
