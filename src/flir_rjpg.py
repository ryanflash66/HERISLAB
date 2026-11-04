"""
Read FLIR-format radiometric JPEGs (R-JPG) without exiftool (HER-82).

A FLIR R-JPG is a normal JPEG preview plus an "FFF" container split across
APP1 segments tagged "FLIR\\0". The container holds:
  - RawData record   : 16-bit sensor counts (stored as TIFF, PNG or plain uint16)
  - CameraInfo record: Planck calibration constants + emissivity etc.

Temperature (degrees C) from raw counts S:
  T = B / ln( R1 / (R2 * (S + O)) + F ) - 273.15
(no atmosphere/emissivity correction; Carter's set uses emissivity 1, distance 0)

Offsets follow ExifTool's FLIR.pm.
"""
import io
import struct
import numpy as np
from PIL import Image

REC_RAW, REC_CAMINFO = 0x0001, 0x0020


def _fff_bytes(jpg: bytes) -> bytes:
    parts, i = [], 2
    while i < len(jpg) - 4 and jpg[i] == 0xFF:
        marker = jpg[i + 1]
        if marker == 0xDA:          # start of scan: no more metadata
            break
        length = int.from_bytes(jpg[i + 2:i + 4], "big")
        seg = jpg[i + 4:i + 2 + length]
        if marker == 0xE1 and seg[:5] == b"FLIR\x00":
            parts.append((seg[6], seg[8:]))   # (part index, payload)
        i += 2 + length
    if not parts:
        raise ValueError("no FLIR APP1 segments")
    return b"".join(p for _, p in sorted(parts))


def _records(fff: bytes):
    if fff[:4] != b"FFF\x00":
        raise ValueError("not an FFF container")
    dir_off, n = struct.unpack(">II", fff[0x18:0x20])
    for k in range(n):
        e = fff[dir_off + 32 * k: dir_off + 32 * (k + 1)]
        rtype, _sub, _ver, _id, off, ln = struct.unpack(">HHIIII", e[:20])
        if rtype:
            yield rtype, fff[off:off + ln]


def _endian(rec: bytes) -> str:
    # first word is 2 in the record's own byte order
    return "<" if struct.unpack("<H", rec[:2])[0] == 2 else ">"


def read_rjpg(path):
    """Return (raw uint16 array HxW, calib dict)."""
    jpg = open(path, "rb").read()
    recs = dict(_records(_fff_bytes(jpg)))
    raw_rec, cam = recs[REC_RAW], recs[REC_CAMINFO]

    e = _endian(raw_rec)
    w, h = struct.unpack(e + "HH", raw_rec[2:6])
    data = raw_rec[0x20:]
    if data[:4] in (b"II*\x00", b"MM\x00*") or data[:8] == b"\x89PNG\r\n\x1a\n":
        raw = np.array(Image.open(io.BytesIO(data)))
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            raw = raw.byteswap()    # FLIR stores PNG raw data byte-swapped
    else:
        raw = np.frombuffer(data[:w * h * 2], dtype=e + "u2").reshape(h, w)
    raw = raw.astype(np.uint16)
    if raw.shape != (h, w):
        raise ValueError(f"raw shape {raw.shape} != header {(h, w)}")

    ce = _endian(cam)
    f = lambda off: struct.unpack(ce + "f", cam[off:off + 4])[0]
    calib = {
        "Emissivity": f(0x20), "ObjectDistance": f(0x24),
        "ReflectedTempK": f(0x28), "AtmosphericTempK": f(0x2C),
        "PlanckR1": f(0x58), "PlanckB": f(0x5C), "PlanckF": f(0x60),
        "PlanckO": struct.unpack(ce + "i", cam[0x308:0x30C])[0],
        "PlanckR2": f(0x30C),
    }
    return raw, calib


def raw_to_celsius(raw, c):
    s = raw.astype(np.float64)
    return (c["PlanckB"] / np.log(c["PlanckR1"] / (c["PlanckR2"] * (s + c["PlanckO"])) + c["PlanckF"]) - 273.15).astype(np.float32)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        raw, c = read_rjpg(p)
        t = raw_to_celsius(raw, c)
        print(p.split("/")[-1], raw.shape, raw.min(), raw.max(),
              f"T median {np.median(t):.1f} C, p99 {np.percentile(t, 99):.1f} C",
              {k: round(v, 4) if isinstance(v, float) else v for k, v in c.items()})
