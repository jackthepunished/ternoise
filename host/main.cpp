// ternoise host: noisy 1-spp path tracer + G-buffers + dataset dump.
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "gl_loader.h"
#include "shader_util.h"
#include "quantize.h"
#include "stb_image_write.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

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
    int size = 256, frames = 0, dump_n = 0, ref_spp = 1024, seed = 1058;
    std::string out_dir;
    bool hidden = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--size") && i + 1 < argc) size = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) frames = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump_n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--ref-spp") && i + 1 < argc) ref_spp = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--hidden")) hidden = true;
        else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (size <= 0 || size % 8) { std::fprintf(stderr, "--size must be a positive multiple of 8\n"); return 2; }
    if (dump_n > 0) {
        hidden = true;
        if (out_dir.empty()) { std::fprintf(stderr, "--dump requires --out DIR\n"); return 2; }
        if (ref_spp < 2) { std::fprintf(stderr, "--ref-spp must be >= 2\n"); return 2; }
    }

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

    auto read_rgba = [&](GLuint tex, std::vector<float>& out) {
        out.resize((size_t)size * size * 4);
        gl::BindTexture(GL_TEXTURE_2D, tex);
        gl::GetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, out.data());
    };
    auto read_red = [&](GLuint tex, std::vector<float>& out) {
        out.resize((size_t)size * size);
        gl::BindTexture(GL_TEXTURE_2D, tex);
        gl::GetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, out.data());
    };

    if (dump_n > 0) {
        std::filesystem::create_directories(out_dir);
        stbi_flip_vertically_on_write(1);   // GL row 0 is bottom; previews written top-down
        const int n_px = size * size;
        std::vector<float> buf_rgba, buf_norm, buf_depth;
        std::vector<int8_t> q7((size_t)7 * n_px), q3((size_t)3 * n_px);
        char path[512];
        auto write_bin = [&](const char* p, const int8_t* d, size_t n) {
            FILE* f = std::fopen(p, "wb");
            if (!f || std::fwrite(d, 1, n, f) != n) { std::fprintf(stderr, "write failed: %s\n", p); std::exit(2); }
            std::fclose(f);
        };
        auto write_png_planes3 = [&](const char* p, const int8_t* planes) {
            std::vector<unsigned char> img((size_t)n_px * 3);
            for (int i = 0; i < n_px; ++i)
                for (int c = 0; c < 3; ++c)
                    img[(size_t)i * 3 + c] = (unsigned char)((int)planes[(size_t)c * n_px + i] + 128);
            stbi_write_png(p, size, size, 3, img.data(), size * 3);
        };
        auto write_png_gray = [&](const char* p, const int8_t* plane) {
            std::vector<unsigned char> img((size_t)n_px);
            for (int i = 0; i < n_px; ++i) img[i] = (unsigned char)((int)plane[i] + 128);
            stbi_write_png(p, size, size, 1, img.data(), size);
        };

        gl::UseProgram(prog_cs);
        for (int v = 0; v < dump_n; ++v) {
            std::mt19937 jrng((unsigned)seed * 2654435761u + (unsigned)v);
            auto U = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(jrng); };
            const float cp[3] = {cam_pos[0] + U(-0.35f, 0.35f), cam_pos[1] + U(-0.2f, 0.2f),
                                 cam_pos[2] + U(-0.5f, 0.5f)};
            const float ct[3] = {cam_target[0] + U(-0.2f, 0.2f), cam_target[1] + U(-0.15f, 0.15f),
                                 cam_target[2]};
            gl::Uniform1ui(u_vseed, (unsigned)(v + 1) * 747796405u + (unsigned)seed);
            gl::Uniform3f(u_cpos, cp[0], cp[1], cp[2]);
            gl::Uniform3f(u_ctgt, ct[0], ct[1], ct[2]);

            for (int f = 0; f < ref_spp; ++f) {
                gl::Uniform1ui(u_frame, (unsigned)f);
                gl::DispatchCompute((GLuint)size / 8, (GLuint)size / 8, 1);
                gl::MemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);
                if (f == 0) {   // frame 0: accum holds exactly the 1-spp noisy sample
                    gl::Finish();
                    read_rgba(tex_accum, buf_rgba);
                    read_rgba(tex_normal, buf_norm);
                    read_red(tex_depth, buf_depth);
                    quantize_frame(buf_rgba.data(), buf_norm.data(), buf_depth.data(), n_px, q7.data());
                    std::snprintf(path, sizeof path, "%s/frame_v%04d_noisy.bin", out_dir.c_str(), v);
                    write_bin(path, q7.data(), (size_t)7 * n_px);
                    std::snprintf(path, sizeof path, "%s/frame_v%04d_noisy.png", out_dir.c_str(), v);
                    write_png_planes3(path, q7.data());
                    if (v == 0) {
                        std::snprintf(path, sizeof path, "%s/frame_v0000_normal.png", out_dir.c_str());
                        write_png_planes3(path, q7.data() + (size_t)3 * n_px);
                        std::snprintf(path, sizeof path, "%s/frame_v0000_depth.png", out_dir.c_str());
                        write_png_gray(path, q7.data() + (size_t)6 * n_px);
                    }
                }
            }
            gl::Finish();
            read_rgba(tex_accum, buf_rgba);
            const float inv = 1.0f / (float)ref_spp;
            for (size_t i = 0; i < buf_rgba.size(); ++i) buf_rgba[i] *= inv;
            quantize_rgb(buf_rgba.data(), n_px, q3.data());
            std::snprintf(path, sizeof path, "%s/frame_v%04d_ref.bin", out_dir.c_str(), v);
            write_bin(path, q3.data(), (size_t)3 * n_px);
            std::snprintf(path, sizeof path, "%s/frame_v%04d_ref.png", out_dir.c_str(), v);
            write_png_planes3(path, q3.data());
            std::printf("view %d/%d done\n", v + 1, dump_n);
            std::fflush(stdout);
        }
        std::snprintf(path, sizeof path, "%s/meta.json", out_dir.c_str());
        FILE* mf = std::fopen(path, "w");
        std::fprintf(mf,
            "{\"size\": %d, \"ref_spp\": %d, \"count\": %d, \"seed\": %d, "
            "\"channels\": \"R,G,B,Nx,Ny,Nz,D\", \"hdr_clamp\": [0, 1], \"row0\": \"bottom\"}\n",
            size, ref_spp, dump_n, seed);
        std::fclose(mf);
        glfwDestroyWindow(win);
        glfwTerminate();
        return 0;
    }

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
