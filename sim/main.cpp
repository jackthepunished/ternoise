// golden <case_dir> : run a golden vector case, diff against expected.txt.
#include "ops.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Minimal scanner for our machine-generated params.json (no general JSON parser).
static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string json_type(const std::string& j) {
    auto p = j.find("\"type\"");
    p = j.find('"', j.find(':', p));
    auto q = j.find('"', p + 1);
    return j.substr(p + 1, q - p - 1);
}

static long json_int_after(const std::string& j, const std::string& key, size_t from, size_t* endpos) {
    auto p = j.find("\"" + key + "\"", from);
    if (p == std::string::npos) { std::fprintf(stderr, "missing key %s\n", key.c_str()); std::exit(2); }
    p = j.find(':', p) + 1;
    long v = std::strtol(j.c_str() + p, nullptr, 10);
    if (endpos) *endpos = p;
    return v;
}

static int run_case(const std::string& dir) {
    const std::string j = slurp(dir + "/params.json");
    const std::string type = json_type(j);
    const Tensor x = load_tensor(dir + "/input.txt");
    const Tensor expected = load_tensor(dir + "/expected.txt");
    Tensor got;

    if (type == "conv") {
        const long shift = json_int_after(j, "shift", 0, nullptr);
        const long use_relu = json_int_after(j, "relu", 0, nullptr);
        got = layer(x, load_tensor(dir + "/weights.txt"), load_tensor(dir + "/bias.txt"),
                    (int)shift, use_relu != 0);
    } else if (type == "network") {
        std::vector<Tensor> ws, bs;
        std::vector<int> shifts;
        size_t pos = 0;
        for (int i = 1; i <= 5; ++i) {
            ws.push_back(load_tensor(dir + "/w" + std::to_string(i) + ".txt"));
            bs.push_back(load_tensor(dir + "/b" + std::to_string(i) + ".txt"));
            size_t end = 0;
            shifts.push_back((int)json_int_after(j, "shift", pos, &end));
            pos = end;
        }
        got = network(x, ws, bs, shifts);
    } else {
        std::fprintf(stderr, "unknown case type '%s'\n", type.c_str());
        return 2;
    }

    if (got.dims != expected.dims) {
        std::printf("FAIL %s (shape mismatch)\n", dir.c_str());
        return 1;
    }
    for (size_t i = 0; i < got.data.size(); ++i)
        if (got.data[i] != expected.data[i]) {
            std::printf("FAIL %s (first mismatch at flat index %zu: got %lld want %lld)\n",
                        dir.c_str(), i, (long long)got.data[i], (long long)expected.data[i]);
            return 1;
        }
    std::printf("PASS %s\n", dir.c_str());
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: golden <case_dir>\n"); return 2; }
    std::string dir(argv[1]);
    while (!dir.empty() && dir.back() == '/') dir.pop_back();
    return run_case(dir);
}
