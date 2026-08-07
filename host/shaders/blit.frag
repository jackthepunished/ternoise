#version 430 core
in vec2 uv;
out vec4 frag;
uniform sampler2D tex;
uniform int display_mode;   // 0 tonemap (pow 1/2.2), 1 raw
uniform float inv_spp;      // 1/frame_count when showing accum, else 1.0
void main() {
    vec3 c = texture(tex, uv).rgb * inv_spp;
    if (display_mode == 0) c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
    frag = vec4(c, 1.0);
}
