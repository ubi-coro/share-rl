"""Driver for a Bota MiniONE that streams over its USB virtual COM port.

The sensor streams 37-byte frames on its own, nothing needs to be sent:
``0xAA | status u16 | Fx Fy Fz Tx Ty Tz f32 | timestamp_us u32 | temperature f32 | crc16``,
little-endian, with the CRC-16/X-25 of everything between the header and the CRC.
"""

import logging
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import serial
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)

FRAME_HEADER = 0xAA
FRAME_FORMAT = struct.Struct("<BH6fIfH")
FRAME_SIZE = FRAME_FORMAT.size

BAUDRATE = 460800  # the USB device ignores it, but a port needs one
READ_TIMEOUT_S = 0.05
CONNECT_TIMEOUT_S = 2.0
# How stale the newest frame may be before latest() raises instead of feeding an old reading to the control loop.
MAX_FRAME_AGE_S = 0.25
HISTORY_FRAMES = 1024  # ~9 s at the sensor's ~110 Hz


@dataclass(frozen=True)
class BotaFrame:
    wrench: np.ndarray  # [Fx, Fy, Fz, Tx, Ty, Tz] in N and Nm, sensor frame
    status: int
    timestamp_us: int  # sensor clock
    temperature_c: float
    host_time: float  # time.monotonic() when the frame was parsed


def crc16_x25(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc ^ 0xFFFF


def decode_frame(raw: bytes) -> BotaFrame | None:
    """Parse one frame, or None if the header or CRC is wrong."""
    if len(raw) != FRAME_SIZE or raw[0] != FRAME_HEADER:
        return None
    _, status, *wrench, timestamp_us, temperature_c, crc = FRAME_FORMAT.unpack(raw)
    if crc != crc16_x25(raw[1:-2]):
        return None
    return BotaFrame(np.array(wrench, dtype=float), status, timestamp_us, temperature_c, time.monotonic())


class BotaDriver:
    """Keeps the newest frames of one sensor in a buffer filled by a background thread."""

    def __init__(self, port: str):
        self.port = port
        self._serial: serial.Serial | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frames: deque[BotaFrame] = deque(maxlen=HISTORY_FRAMES)
        self._error: Exception | None = None

    @property
    def is_connected(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"Bota sensor on {self.port} is already connected")
        self._serial = serial.Serial(self.port, BAUDRATE, timeout=READ_TIMEOUT_S)
        self._serial.reset_input_buffer()
        self._stop.clear()
        self._error = None
        self._frames.clear()
        self._thread = threading.Thread(target=self._read_loop, name="bota-reader", daemon=True)
        self._thread.start()

        deadline = time.monotonic() + CONNECT_TIMEOUT_S
        while not self._frames:
            if time.monotonic() > deadline or not self._thread.is_alive():
                self.disconnect()
                raise ConnectionError(
                    f"No valid Bota frames on {self.port} within {CONNECT_TIMEOUT_S}s: "
                    "check the port, and that no other program holds it open."
                )
            time.sleep(0.01)
        logger.info(f"Bota sensor connected on {self.port}")

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def latest(self) -> BotaFrame:
        """The newest frame. Raises if the stream has stalled or the port died."""
        if self._error is not None:
            raise ConnectionError(f"Bota sensor on {self.port} failed: {self._error!r}") from self._error
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Bota sensor on {self.port} is not connected")
        with self._lock:
            frame = self._frames[-1] if self._frames else None
        if frame is None or time.monotonic() - frame.host_time > MAX_FRAME_AGE_S:
            raise TimeoutError(f"Bota sensor on {self.port} stopped streaming (newest frame older than {MAX_FRAME_AGE_S}s)")
        return frame

    def recent(self, duration_s: float) -> np.ndarray:
        """Wrenches of the frames received in the last ``duration_s`` seconds, shape (N, 6)."""
        oldest = time.monotonic() - duration_s
        with self._lock:
            wrenches = [frame.wrench for frame in self._frames if frame.host_time >= oldest]
        return np.array(wrenches).reshape(-1, 6)

    def _read_loop(self) -> None:
        buffer = bytearray()
        try:
            while not self._stop.is_set():
                chunk = self._serial.read(self._serial.in_waiting or 1)
                if chunk:
                    buffer += chunk
                    self._extract_frames(buffer)
        except (serial.SerialException, OSError) as error:
            self._error = error

    def _extract_frames(self, buffer: bytearray) -> None:
        """Consume complete frames from the front of ``buffer``, resyncing on the header after a bad one."""
        while len(buffer) >= FRAME_SIZE:
            frame = decode_frame(bytes(buffer[:FRAME_SIZE]))
            if frame is None:
                next_header = buffer.find(FRAME_HEADER, 1)
                del buffer[: len(buffer) if next_header == -1 else next_header]
                continue
            del buffer[:FRAME_SIZE]
            with self._lock:
                self._frames.append(frame)
