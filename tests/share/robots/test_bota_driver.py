"""Frame decoding and resync of the Bota stream, including a frame captured from a real MiniONE."""

from __future__ import annotations

import struct

import pytest
from lerobot.utils.errors import DeviceNotConnectedError

from share.robots.viperx.lerobot_robot_viperx.bota.driver import FRAME_SIZE, BotaDriver, crc16_x25, decode_frame

# Frame read from a MiniONE on /dev/ttyACM*: validates the CRC-16/X-25 choice against the hardware.
REAL_FRAME = bytes.fromhex(
    "aa 00 00 14 d4 f1 41 7f 63 c2 42 25 da 2f c3 b0 45 ea 3f c2 12 c7 be e6 55 5d 3d cd fa f2 fe e4 0e fb 41 72 26"
)


def _frame(wrench=(1.0, 2.0, 3.0, 0.1, 0.2, 0.3), status=0, timestamp_us=1000, temperature_c=31.0) -> bytes:
    body = struct.pack("<H6fIf", status, *wrench, timestamp_us, temperature_c)
    return bytes([0xAA]) + body + struct.pack("<H", crc16_x25(body))


def test_real_frame_decodes():
    frame = decode_frame(REAL_FRAME)

    assert frame is not None
    assert frame.wrench[:3].tolist() == pytest.approx([30.229, 97.194, -175.852], abs=0.01)
    assert frame.status == 0
    assert frame.temperature_c == pytest.approx(31.38, abs=0.01)


def test_decode_rejects_a_corrupted_payload_and_a_wrong_header():
    corrupted = bytearray(REAL_FRAME)
    corrupted[10] ^= 0xFF

    assert decode_frame(bytes(corrupted)) is None
    assert decode_frame(b"\x00" + REAL_FRAME[1:]) is None
    assert decode_frame(REAL_FRAME[:-1]) is None


def test_stream_resyncs_after_garbage_and_a_bad_frame():
    sensor = BotaDriver("unused")
    bad = bytearray(_frame(wrench=(9.0,) * 6))
    bad[5] ^= 0xFF
    buffer = bytearray(b"\x01\x02\xaa\x03" + _frame(wrench=(1.0,) * 6) + bytes(bad) + _frame(wrench=(2.0,) * 6))
    buffer += _frame()[: FRAME_SIZE // 2]  # a frame still arriving

    sensor._extract_frames(buffer)

    assert sensor.recent(10.0)[:, 0].tolist() == [1.0, 2.0]
    assert len(buffer) == FRAME_SIZE // 2  # the partial frame waits for its remaining bytes


def test_latest_requires_a_connection():
    with pytest.raises(DeviceNotConnectedError):
        BotaDriver("unused").latest()
