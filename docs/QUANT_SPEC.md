# ternoise Quantization Contract (QUANT_SPEC)

Every implementation (Python reference, PyTorch eval path, C++ golden model,
RTL) MUST produce bit-identical results to this document. Ambiguity is a bug.

## 1. Weight ternarization (offline, absmean / BitNet b1.58)

Given a layer's FP32 master weights W:

    gamma = mean(|W|)                (over the whole layer)
    Wt    = clamp(round_half_away(W / gamma), -1, +1)   if gamma > 0
    Wt    = 0                                           if gamma == 0

Example: W = [0.4, -0.05, 0.3] -> gamma = 0.25 -> W/gamma = [1.6, -0.2, 1.2]
-> round [2, 0, 1] -> clamp [+1, 0, +1].

gamma itself never appears at inference time; its effect is absorbed by the
per-layer requant shift chosen at calibration (M3).

## 2. Activations

INT8, range [-128, 127]. Input encodings from the renderer:

    color   c in [0,1]:  q = clamp(round(c * 255) - 128, -128, 127)
    normal  n in [-1,1]: q = clamp(round(n * 127), -128, 127)
    depth   d in [0,1]:  q = clamp(round(d * 255) - 128, -128, 127)

Input layout: 7 channels = [R, G, B, Nx, Ny, Nz, D].
Decode of the final RGB: c = (q + 128) / 255.

## 3. Convolution layer (the only compute primitive)

For each output channel co at each pixel (y, x):

    acc = bias[co] + sum over (ci, ky, kx in 3x3) of
          Wt[co,ci,ky,kx] * X[ci, y+ky-1, x+kx-1]

- Border: zero padding (contributes value 0), output HxW equals input HxW.
- Wt in {-1,0,+1}: the product is an add, a subtract, or nothing.
- bias is a per-output-channel int32 constant (derivation from FP bias is an
  M3 calibration concern; it is a plain int32 add here).
- acc range for C_in channels: |acc| <= 9 * C_in * 128 + |bias|.
  Accumulator width rule (RTL): ceil(log2(9 * C_in * 128)) + 1 bits, plus
  headroom for bias. Software uses int64 and asserts the int32 range.

## 4. Requantization (acc -> int8)

    requant(acc, s) = clamp(rha(acc, s), -128, 127),  s in [0, 15]
    rha(acc, 0) = acc
    rha(acc, s) = (acc + h) >> s          if acc >= 0, h = 1 << (s-1)
                = -(((-acc) + h) >> s)    if acc <  0

(">>" is a logical shift of a non-negative number; this realizes
round-half-away-from-zero with no multiplier and no division.)

Normative examples (test vectors):

    acc     s  ->  y
    0       0      0
    127     0      127
    200     0      127   (clamp)
    -200    0      -128  (clamp)
    5       1      3     (2.5 away from zero)
    -5      1      -3
    3       1      2     (1.5 -> 2)
    -3      1      -2
    2       2      1     (0.5 -> 1)
    -2      2      -1
    36864   5      127   (1152 pre-clamp)

## 5. Nonlinearity and network v1

    relu(q) = max(0, q), applied AFTER requant, on int8.

Network v1: five 3x3 layers, channels 7->16->16->16->16->3.
Layers 1-4: conv -> requant(s_i) -> relu. Layer 5: conv -> requant(s_5), no relu.
Residual head: out_rgb[ch] = clamp(x_in[ch] + y5[ch], -128, 127) for ch in 0..2,
where x_in is the network INPUT (channels 0-2 = noisy RGB).

## 6. Packed weight format (2 bits per weight)

Flattening order: (c_out, c_in, ky, kx), row-major (kx fastest).
Encoding: 00 = 0, 01 = +1, 10 = -1, 11 = ILLEGAL (loader must error).
Byte packing: weight i lives in byte i//4, bits [2*(i%4)+1 : 2*(i%4)]
(first weight in the least significant bits). Final partial byte zero-padded.
.mem format: one byte per line as two lowercase hex chars, for $readmemh.
Biases: int32, .mem as eight lowercase hex chars per line (two's complement),
one line per output channel, layer files kept separate.

## 7. Golden vector file formats

Tensor text file: first line = rank then dims (e.g. "3 8 16 16"); following
whitespace-separated ints, row-major. params.json per case:

    conv case:    {"type": "conv", "shift": S, "relu": 0|1}
                  files: input.txt, weights.txt, bias.txt, expected.txt
    network case: {"type": "network",
                   "layers": [{"shift": S1, "relu": 1}, ..., {"shift": S5, "relu": 0}]}
                  files: input.txt, w1..w5.txt, b1..b5.txt, expected.txt
                  expected.txt is the post-residual 3-channel RGB output.

Determinism: generator seed is 1058. Regenerating must be byte-identical.
