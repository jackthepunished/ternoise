import numpy as np
from model.ref_ops import requant, conv3x3, relu, layer, residual_add

SPEC_REQUANT = [  # (acc, shift, expected) — verbatim from QUANT_SPEC section 4
    (0, 0, 0), (127, 0, 127), (200, 0, 127), (-200, 0, -128),
    (5, 1, 3), (-5, 1, -3), (3, 1, 2), (-3, 1, -2),
    (2, 2, 1), (-2, 2, -1), (36864, 5, 127),
]

def test_requant_spec_table():
    for (a, s, e) in SPEC_REQUANT:
        assert requant(np.array([a], dtype=np.int64), s)[0] == e, (a, s)

def test_conv3x3_hand_computed_pixel():
    # 1 channel 4x4 input, values 1..16 row-major
    x = np.arange(1, 17, dtype=np.int64).reshape(1, 4, 4)
    # vertical-edge kernel: +1 col, 0 col, -1 col
    w = np.array([[[[1, 0, -1]] * 3]], dtype=np.int64)  # (1,1,3,3)
    b = np.array([10], dtype=np.int64)
    acc = conv3x3(x, w, b)
    # interior (1,1): +(1+5+9) - (3+7+11) + 10 = 15 - 21 + 10 = 4
    assert acc[0, 1, 1] == 4
    # border (0,0), zero-padded: -(x[0,1] + x[1,1]) + 10 = -(2+6) + 10 = 2
    assert acc[0, 0, 0] == 2

def test_layer_applies_requant_and_relu():
    x = np.arange(1, 17, dtype=np.int64).reshape(1, 4, 4)
    w = np.array([[[[1, 0, -1]] * 3]], dtype=np.int64)
    b = np.array([10], dtype=np.int64)
    y = layer(x, w, b, shift=2, use_relu=True)
    assert y[0, 1, 1] == 1          # requant(4, 2) = (4+2)>>2 = 1
    assert y[0, 0, 0] == 1          # requant(2, 2) = (2+2)>>2 = 1
    assert y.min() >= 0             # relu

def test_relu_and_residual_clamp():
    assert relu(np.array([-5, 0, 7], dtype=np.int64)).tolist() == [0, 0, 7]
    r = residual_add(np.array([120], dtype=np.int64), np.array([100], dtype=np.int64))
    assert r[0] == 127
    r = residual_add(np.array([-120], dtype=np.int64), np.array([-100], dtype=np.int64))
    assert r[0] == -128

def test_accumulator_extreme_no_overflow():
    # all-(+1) weights, all-127 input, C_in=32: acc = 9*32*127 per pixel
    x = np.full((32, 5, 5), 127, dtype=np.int64)
    w = np.ones((1, 32, 3, 3), dtype=np.int64)
    b = np.zeros(1, dtype=np.int64)
    acc = conv3x3(x, w, b)
    assert acc[0, 2, 2] == 9 * 32 * 127
