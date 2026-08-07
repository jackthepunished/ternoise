import numpy as np
import pytest
from tools.pack_weights import pack, unpack, to_mem, bias_to_mem

def test_known_byte():
    # weights [+1, -1, 0, +1] -> bits: w0=01, w1=10, w2=00, w3=01
    # byte = 01_00_10_01 (w3..w0) = 0x49
    w = np.array([1, -1, 0, 1], dtype=np.int64)
    assert pack(w) == bytes([0x49])

def test_roundtrip_random():
    rng = np.random.default_rng(1058)
    w = rng.integers(-1, 2, size=(16, 7, 3, 3)).astype(np.int64)
    assert np.array_equal(unpack(pack(w), w.shape), w)

def test_partial_byte_zero_padded():
    w = np.array([1], dtype=np.int64)          # 1 weight -> 1 byte, upper bits 0
    assert pack(w) == bytes([0x01])

def test_illegal_encoding_rejected():
    with pytest.raises(ValueError):
        unpack(bytes([0b00000011]), (1,))       # '11' pattern

def test_mem_formats():
    assert to_mem(bytes([0x49, 0x01])) == "49\n01\n"
    assert bias_to_mem(np.array([255, -1], dtype=np.int64)) == "000000ff\nffffffff\n"
