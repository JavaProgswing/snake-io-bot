"""Minimal scrcpy 4.0 client: H264 video frames + touch injection over the
scrcpy server's video/control sockets. No scrcpy.exe window needed — we push
and launch the bundled scrcpy-server jar ourselves via adb.
"""

import random
import socket
import struct
import subprocess
import threading
import time

import av
import numpy as np

ACTION_DOWN = 0
ACTION_UP = 1
ACTION_MOVE = 2

CONTROL_MSG_INJECT_TOUCH = 2
CONTROL_MSG_INJECT_KEYCODE = 0

KEYCODE_BACK = 4

PRESSURE_MAX = 0xFFFF


class ScrcpyLink:
    def __init__(self, adb, server_jar, version="4.0", port=27183,
                 max_size=1184, max_fps=30, video_bit_rate=4_000_000,
                 log_path=None):
        self.adb = adb
        self.server_jar = server_jar
        self.version = version
        self.port = port
        self.max_size = max_size
        self.max_fps = max_fps
        self.video_bit_rate = video_bit_rate
        self.log_path = log_path

        self._server_proc = None
        self._video_sock = None
        self._control_sock = None
        self._decode_thread = None
        self._running = False

        self._frame_lock = threading.Lock()
        self._frame = None          # np BGR
        self._frame_seq = 0

        self._ctrl_lock = threading.Lock()

    # ---------- lifecycle ----------

    def _adb(self, *args, **kw):
        return subprocess.run([self.adb, *args], capture_output=True,
                              text=True, timeout=20, **kw)

    def start(self):
        # scid parsed as signed 31-bit hex int on the server: keep MSB clear
        scid = random.choice("01234567") + \
            "".join(random.choice("0123456789abcdef") for _ in range(7))

        r = self._adb("push", self.server_jar, "/data/local/tmp/scrcpy-server.jar")
        if r.returncode != 0:
            raise RuntimeError(f"adb push failed: {r.stderr}")

        self._adb("forward", f"tcp:{self.port}", f"localabstract:scrcpy_{scid}")

        log = open(self.log_path, "wb") if self.log_path else subprocess.DEVNULL
        self._server_proc = subprocess.Popen(
            [self.adb, "shell",
             "CLASSPATH=/data/local/tmp/scrcpy-server.jar",
             "app_process", "/", "com.genymobile.scrcpy.Server",
             self.version,
             f"scid={scid}",
             "log_level=info",
             "video=true", "audio=false", "control=true",
             "tunnel_forward=true",
             "video_codec=h264",
             f"max_size={self.max_size}",
             f"max_fps={self.max_fps}",
             f"video_bit_rate={self.video_bit_rate}",
             "send_device_meta=false",
             "send_frame_meta=false",
             "send_dummy_byte=true",
             "stay_awake=true",
             "cleanup=true"],
            stdout=log, stderr=subprocess.STDOUT)

        # adb forward accepts TCP even before the server binds its abstract
        # socket — in that case the connection just EOFs. Retry until the
        # dummy byte actually arrives.
        t0 = time.time()
        while True:
            s = self._connect_retry()
            s.settimeout(2)
            try:
                dummy = s.recv(1)
            except socket.timeout:
                dummy = b""
            if dummy == b"\x00":
                s.settimeout(None)
                self._video_sock = s
                break
            s.close()
            if self._server_proc.poll() is not None:
                raise RuntimeError("scrcpy server exited early (see log)")
            if time.time() - t0 > 15:
                raise RuntimeError(f"no dummy byte from server (got {dummy!r})")
            time.sleep(0.25)
        self._control_sock = self._connect_retry()

        # scrcpy 4.0 always sends 12 bytes of codec metadata on the video
        # socket (codec fourcc + width + height) — consume if present.
        head = self._recv_exact(self._video_sock, 4)
        if head in (b"h264", b"h265", b"av01"):
            self._recv_exact(self._video_sock, 8)
            self._preroll = b""
        else:
            self._preroll = head

        self._running = True
        self.ctrl_ok = True
        self._decode_thread = threading.Thread(target=self._decode_loop, daemon=True)
        self._decode_thread.start()

        # wait for first frame
        t0 = time.time()
        while time.time() - t0 < 10:
            if self.frame()[0] is not None:
                return
            time.sleep(0.05)
        raise RuntimeError("no video frame within 10s (check server log)")

    def _connect_retry(self, timeout=10.0):
        t0 = time.time()
        while True:
            try:
                s = socket.create_connection(("127.0.0.1", self.port), timeout=2)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return s
            except OSError:
                if self._server_proc.poll() is not None:
                    raise RuntimeError("scrcpy server exited early (see log)")
                if time.time() - t0 > timeout:
                    raise
                time.sleep(0.2)

    def stop(self):
        self._running = False
        for s in (self._video_sock, self._control_sock):
            try:
                if s:
                    s.close()
            except OSError:
                pass
        if self._server_proc:
            self._server_proc.terminate()
        try:
            self._adb("forward", "--remove", f"tcp:{self.port}")
        except Exception:
            pass

    # ---------- video ----------

    @staticmethod
    def _recv_exact(sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise RuntimeError("video socket closed during handshake")
            buf += chunk
        return buf

    def _decode_loop(self):
        codec = av.CodecContext.create("h264", "r")
        pending = self._preroll
        while self._running:
            try:
                data = pending + self._video_sock.recv(0x20000)
                pending = b""
            except OSError:
                break
            if not data:
                break
            try:
                for packet in codec.parse(data):
                    for frame in codec.decode(packet):
                        arr = frame.to_ndarray(format="bgr24")
                        with self._frame_lock:
                            self._frame = arr
                            self._frame_seq += 1
            except av.error.InvalidDataError:
                continue

    def frame(self):
        """Returns (bgr_array_or_None, seq)."""
        with self._frame_lock:
            if self._frame is None:
                return None, 0
            return self._frame, self._frame_seq

    # ---------- control ----------

    def _send(self, payload):
        with self._ctrl_lock:
            try:
                self._control_sock.sendall(payload)
            except OSError:
                # control socket dropped (phone/USB hiccup). Flag it instead of
                # crashing -- the bot loop checks ctrl_ok and reconnects.
                self.ctrl_ok = False

    def touch(self, action, x, y, pointer_id=1):
        """x, y in the video-frame coordinate space."""
        frame, _ = self.frame()
        if frame is None:
            return
        h, w = frame.shape[:2]
        x = int(max(0, min(w - 1, x)))
        y = int(max(0, min(h - 1, y)))
        pressure = 0 if action == ACTION_UP else PRESSURE_MAX
        msg = struct.pack(">BBqiiHHHii",
                          CONTROL_MSG_INJECT_TOUCH, action, pointer_id,
                          x, y, w, h, pressure, 0, 0)
        self._send(msg)

    def key(self, keycode):
        # keycode down + up
        for action in (0, 1):
            msg = struct.pack(">BBiii", CONTROL_MSG_INJECT_KEYCODE,
                              action, keycode, 0, 0)
            self._send(msg)

    def launch_app(self, package):
        """Bring the game to the foreground — recovers from crashes, the
        system launcher, or any screen the state machine can't act on."""
        self._adb("shell", "monkey", "-p", package,
                  "-c", "android.intent.category.LAUNCHER", "1")

    def tap(self, x, y, pointer_id=4, hold=0.06):
        self.touch(ACTION_DOWN, x, y, pointer_id)
        time.sleep(hold)
        self.touch(ACTION_UP, x, y, pointer_id)

    def clear_pointers(self):
        """Heal InputDispatcher's stuck-pointer state left by a killed
        session. A held-down virtual pointer makes the dispatcher drop ALL
        injected touches as 'inconsistent'. Cure: press 4 fingers — each
        POINTER_DOWN that consistently EXTENDS the stuck set is accepted,
        growing it to {0..3} — then lift them in order, unwinding the whole
        set to empty. Harmless corner touches when nothing was stuck."""
        pids = (90, 91, 92, 93)
        coords = ((2, 2), (8, 2), (2, 8), (8, 8))
        for pid, (x, y) in zip(pids, coords):
            self.touch(ACTION_DOWN, x, y, pid)
            time.sleep(0.03)
        for pid, (x, y) in zip(pids, coords):
            self.touch(ACTION_UP, x, y, pid)
            time.sleep(0.03)
