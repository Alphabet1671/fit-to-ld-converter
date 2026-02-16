"""
fit_to_ld_gui.py  (PyQt6)

GUI tool to batch-convert Garmin .fit -> MoTeC .ld.

Pipeline (per file):
  1) Use Garmin FitCSVTool.jar (-b) to decode FIT -> CSV(s)
  2) Parse FitCSVTool "message-stream" CSV into channels (keeps all fields)
  3) Resample channels to fixed frequency
  4) Write MoTeC .ld (LD writer code adapted from gotzl/ldparser, GPL-3.0)

Batch behavior:
  - Input: a folder containing .fit files (non-recursive)
  - Output: a folder; each output .ld filename matches the .fit basename
  - Skips any .fit whose .ld already exists in the output folder

New:
  - If the FIT filename matches "YYYY-MM-DD-HH-MM-SS.fit", that timestamp is used
    as the Date/Time in the MoTeC LD header. Otherwise falls back to current time.

Requirements:
  - Python 3.9+
  - Java (to run FitCSVTool.jar)
  - pip install PyQt6 numpy pandas python-dateutil
"""

from __future__ import annotations

import re
import sys
import json
import time
import shutil
import subprocess
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QSignalBlocker
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QDoubleSpinBox,
    QWidget,
    QVBoxLayout,
    QComboBox,
)

# =============================================================================
# Minimal MoTeC LD writer (ADAPTED FROM gotzl/ldparser.py, GPL-3.0)
# =============================================================================
import datetime
import struct


# -------------------------------
# NEW: parse date/time from filename
# -------------------------------
FNAME_DT_RE = re.compile(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})-(?P<H>\d{2})-(?P<M>\d{2})-(?P<S>\d{2})")


def parse_datetime_from_fit_filename(path: Path) -> Optional[datetime.datetime]:
    """
    Parse timestamp from filename like:
      YYYY-MM-DD-HH-MM-SS.fit
    Returns a naive datetime in local time (MoTeC header stores date/time strings).
    """
    m = FNAME_DT_RE.search(path.stem)
    if not m:
        return None
    try:
        return datetime.datetime(
            int(m.group("y")),
            int(m.group("m")),
            int(m.group("d")),
            int(m.group("H")),
            int(m.group("M")),
            int(m.group("S")),
        )
    except Exception:
        return None


class ldEvent:
    fmt = "<64s64s1024sH"

    def __init__(self, name: str, session: str, comment: str, venue_ptr: int, venue):
        self.name = name
        self.session = session
        self.comment = comment
        self.venue_ptr = venue_ptr
        self.venue = venue

    def write(self, f):
        f.write(
            struct.pack(
                ldEvent.fmt,
                self.name.encode(),
                self.session.encode(),
                self.comment.encode(),
                self.venue_ptr,
            )
        )
        if self.venue_ptr > 0 and self.venue is not None:
            f.seek(self.venue_ptr)
            self.venue.write(f)


class ldVenue:
    fmt = "<64s1034xH"

    def __init__(self, name: str, vehicle_ptr: int, vehicle):
        self.name = name
        self.vehicle_ptr = vehicle_ptr
        self.vehicle = vehicle

    def write(self, f):
        f.write(struct.pack(ldVenue.fmt, self.name.encode(), self.vehicle_ptr))
        if self.vehicle_ptr > 0 and self.vehicle is not None:
            f.seek(self.vehicle_ptr)
            self.vehicle.write(f)


class ldVehicle:
    fmt = "<64s128xI32s32s"

    def __init__(self, vid: str, weight: int, vtype: str, comment: str):
        self.id = vid
        self.weight = weight
        self.type = vtype
        self.comment = comment

    def write(self, f):
        f.write(
            struct.pack(
                ldVehicle.fmt,
                self.id.encode(),
                int(self.weight),
                self.type.encode(),
                self.comment.encode(),
            )
        )


class ldHead:
    fmt = (
        "<"
        + (
            "I4x"  # marker
            "II"  # meta_ptr data_ptr
            "20x"
            "I"  # event_ptr
            "24x"
            "HHH"
            "I"  # device serial
            "8s"  # device type
            "H"  # device version
            "H"
            "I"  # num channels
            "4x"
            "16s"  # date
            "16x"
            "16s"  # time
            "16x"
            "64s"  # driver
            "64s"  # vehicle id
            "64x"
            "64s"  # venue
            "64x"
            "1024x"
            "I"  # pro-enable magic
            "66x"
            "64s"  # short comment
            "126x"
        )
    )

    def __init__(
        self,
        meta_ptr: int,
        data_ptr: int,
        event_ptr: int,
        event: ldEvent,
        driver: str,
        vehicleid: str,
        venue: str,
        dt: datetime.datetime,
        short_comment: str,
    ):
        self.meta_ptr = meta_ptr
        self.data_ptr = data_ptr
        self.event_ptr = event_ptr
        self.event = event
        self.driver = driver
        self.vehicleid = vehicleid
        self.venue = venue
        self.datetime = dt
        self.short_comment = short_comment

    def write(self, f, n_channels: int):
        f.write(
            struct.pack(
                ldHead.fmt,
                0x40,
                self.meta_ptr,
                self.data_ptr,
                self.event_ptr,
                1,
                0x4240,
                0x0F,
                0x1F44,
                b"ADL",
                420,
                0xADB0,
                n_channels,
                self.datetime.date().strftime("%d/%m/%Y").encode(),
                self.datetime.time().strftime("%H:%M:%S").encode(),
                self.driver.encode(),
                self.vehicleid.encode(),
                self.venue.encode(),
                0xC81A4,  # pro enable magic
                self.short_comment.encode(),
            )
        )
        if self.event_ptr > 0 and self.event is not None:
            f.seek(self.event_ptr)
            self.event.write(f)


class ldChan:
    fmt = (
        "<"
        + (
            "IIII"  # prev next data_ptr n_data
            "H"  # counter
            "HHH"  # dtype_a dtype freq
            "hhhh"  # shift mul scale dec
            "32s"  # name
            "8s"  # short name
            "12s"  # unit
            "40x"
        )
    )

    def __init__(
        self,
        meta_ptr: int,
        prev_meta_ptr: int,
        next_meta_ptr: int,
        data_ptr: int,
        data_len: int,
        dtype: Any,
        freq: int,
        shift: int,
        mul: int,
        scale: int,
        dec: int,
        name: str,
        short_name: str,
        unit: str,
        data: np.ndarray,
    ):
        self.meta_ptr = meta_ptr
        self.prev_meta_ptr = prev_meta_ptr
        self.next_meta_ptr = next_meta_ptr
        self.data_ptr = data_ptr
        self.data_len = data_len
        self.dtype = np.dtype(dtype)
        self.freq = int(freq)
        self.shift = int(shift)
        self.mul = int(mul)
        self.scale = int(scale)
        self.dec = int(dec)
        self.name = name[:32]
        self.short_name = short_name[:8]
        self.unit = unit[:12]
        self._data = data

    @property
    def data(self) -> np.ndarray:
        return self._data

    def write(self, f, n: int):
        if self.dtype == np.dtype(np.float16) or self.dtype == np.dtype(np.float32):
            dtype_a = 0x07
            dtype = {np.dtype(np.float16).type: 2, np.dtype(np.float32).type: 4}[self.dtype.type]
        else:
            dtype_a = 0x05 if self.dtype == np.dtype(np.int32) else 0x03
            dtype = {np.dtype(np.int16).type: 2, np.dtype(np.int32).type: 4}[self.dtype.type]

        f.write(
            struct.pack(
                ldChan.fmt,
                self.prev_meta_ptr,
                self.next_meta_ptr,
                self.data_ptr,
                self.data_len,
                0x2EE1 + n,
                dtype_a,
                dtype,
                self.freq,
                self.shift,
                self.mul,
                self.scale,
                self.dec,
                self.name.encode(),
                self.short_name.encode(),
                self.unit.encode(),
            )
        )


class ldData:
    def __init__(self, head: ldHead, channs: List[ldChan]):
        self.head = head
        self.channs = channs

    def write(self, fpath: str):
        def conv(c: ldChan) -> np.ndarray:
            return ((c.data / c.mul) - c.shift) * c.scale / pow(10.0, -c.dec)

        with open(fpath, "wb") as f:
            self.head.write(f, len(self.channs))
            f.seek(self.channs[0].meta_ptr)
            for i, c in enumerate(self.channs):
                c.write(f, i)

            for c in self.channs:
                f.write(conv(c).astype(c.dtype, copy=False).tobytes(order="C"))


def build_ld_from_dataframe(
    df: pd.DataFrame,
    out_ld_path: str,
    sample_hz: float,
    meta: Dict[str, str],
    units: Dict[str, str],
    dt_for_header: Optional[datetime.datetime] = None,  # NEW
):
    df_num = df.copy()
    for c in df_num.columns:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    df_num = df_num.astype(np.float32)

    head_size = struct.calcsize(ldHead.fmt)
    event_size = struct.calcsize(ldEvent.fmt)
    venue_size = struct.calcsize(ldVenue.fmt)
    vehicle_size = struct.calcsize(ldVehicle.fmt)
    chan_meta_size = struct.calcsize(ldChan.fmt)

    n_ch = len(df_num.columns)
    event_ptr = head_size
    venue_ptr = event_ptr + event_size
    vehicle_ptr = venue_ptr + venue_size
    meta_ptr0 = vehicle_ptr + vehicle_size
    data_ptr0 = meta_ptr0 + n_ch * chan_meta_size

    vehicle = ldVehicle(
        vid=meta.get("vehicleid", "vehicle"),
        weight=int(float(meta.get("vehicle_weight", "0") or "0")),
        vtype=meta.get("vehicle_type", ""),
        comment=meta.get("vehicle_comment", ""),
    )
    venue = ldVenue(name=meta.get("venue", "venue"), vehicle_ptr=vehicle_ptr, vehicle=vehicle)
    event = ldEvent(
        name=meta.get("event_name", "event"),
        session=meta.get("event_session", "0"),
        comment=meta.get("event_comment", ""),
        venue_ptr=venue_ptr,
        venue=venue,
    )

    head = ldHead(
        meta_ptr=meta_ptr0,
        data_ptr=data_ptr0,
        event_ptr=event_ptr,
        event=event,
        driver=meta.get("driver", "driver"),
        vehicleid=meta.get("vehicleid", "vehicle"),
        venue=meta.get("venue", "venue"),
        dt=dt_for_header if dt_for_header is not None else datetime.datetime.now(),  # NEW
        short_comment=meta.get("short_comment", ""),
    )

    channs: List[ldChan] = []
    meta_ptr = meta_ptr0
    data_ptr = data_ptr0
    prev_ptr = 0

    for i, col in enumerate(df_num.columns):
        next_ptr = meta_ptr + chan_meta_size if i < n_ch - 1 else 0
        data = df_num[col].to_numpy(dtype=np.float32, copy=False)

        unit = units.get(col, "")
        short = re.sub(r"[^A-Za-z0-9]", "", col)[:8] or f"C{i:03d}"

        chan = ldChan(
            meta_ptr=meta_ptr,
            prev_meta_ptr=prev_ptr,
            next_meta_ptr=next_ptr,
            data_ptr=data_ptr,
            data_len=len(data),
            dtype=np.float32,
            freq=int(round(sample_hz)),
            shift=0,
            mul=1,
            scale=1,
            dec=0,
            name=col,
            short_name=short,
            unit=unit,
            data=data,
        )

        channs.append(chan)

        prev_ptr = meta_ptr
        meta_ptr = next_ptr if next_ptr != 0 else meta_ptr + chan_meta_size
        data_ptr += data.nbytes

    ld = ldData(head, channs)
    ld.write(out_ld_path)


# =============================================================================
# FIT CSV parsing (FitCSVTool message-stream CSV)
# =============================================================================

def is_fitcsv_message_stream(df: pd.DataFrame) -> bool:
    cols = {c.strip().lower() for c in df.columns}
    return (
        "type" in cols
        and "message" in cols
        and any(c.startswith("field") for c in cols)
        and any(c.startswith("value") for c in cols)
    )


def _max_field_index(cols: List[str]) -> int:
    m = 0
    for c in cols:
        mm = re.match(r"field\s*(\d+)$", c.strip().lower())
        if mm:
            m = max(m, int(mm.group(1)))
    return m


def _extract_pairs_from_row(row: pd.Series, max_i: int) -> Dict[str, Tuple[Any, Any]]:
    out: Dict[str, Tuple[Any, Any]] = {}
    for i in range(1, max_i + 1):
        f = row.get(f"Field {i}")
        if pd.isna(f):
            continue
        v = row.get(f"Value {i}")
        u = row.get(f"Units {i}")
        out[str(f)] = (v, u)
    return out


def _fit_timestamp_to_unix_seconds(v: Any, u: Any) -> Optional[float]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        x = float(v)
    except Exception:
        return None

    uu = ("" if u is None else str(u)).strip().lower()
    fit_epoch_unix = datetime.datetime(1989, 12, 31, tzinfo=datetime.timezone.utc).timestamp()
    if "ms" in uu:
        return fit_epoch_unix + x / 1000.0
    return fit_epoch_unix + x


def resample_channel(
    t_raw: np.ndarray,
    y_raw: np.ndarray,
    t_grid: np.ndarray,
    method: str,
) -> np.ndarray:
    mask = np.isfinite(t_raw) & np.isfinite(y_raw)
    t = t_raw[mask]
    y = y_raw[mask]
    if t.size == 0:
        return np.full_like(t_grid, np.nan, dtype=np.float32)

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    _, idx_last = np.unique(t[::-1], return_index=True)
    keep = (t.size - 1 - idx_last)
    keep.sort()
    t = t[keep]
    y = y[keep]

    if method == "linear":
        return np.interp(t_grid, t, y, left=np.nan, right=np.nan).astype(np.float32)

    pos = np.searchsorted(t, t_grid, side="right") - 1
    out = np.full_like(t_grid, np.nan, dtype=np.float32)
    good = pos >= 0
    out[good] = y[pos[good]].astype(np.float32, copy=False)
    return out


def load_all_fitcsv_outputs(prefix_no_ext: Path) -> List[Path]:
    parent = prefix_no_ext.parent
    stem = prefix_no_ext.name
    return sorted(parent.glob(stem + "*.csv"))


def fit_to_dataframe_via_fitcsvtool(
    fit_path: Path,
    fitcsvtool_jar: Path,
    temp_dir: Path,
    sample_hz: float,
    resample_mode: str,
    log_cb,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Dict[int, str]]]:

    if shutil.which("java") is None:
        raise RuntimeError("Java not found on PATH. Install a JRE/JDK so `java` works.")

    temp_dir.mkdir(parents=True, exist_ok=True)
    prefix = temp_dir / fit_path.stem

    log_cb(f"Running FitCSVTool: {fit_path.name}")
    cmd = ["java", "-jar", str(fitcsvtool_jar), "-b", str(fit_path), str(prefix)]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError("FitCSVTool failed.\n\nSTDOUT:\n" + proc.stdout + "\n\nSTDERR:\n" + proc.stderr)

    log_cb(proc.stdout.strip() or "FitCSVTool finished.")

    csv_files = load_all_fitcsv_outputs(prefix)
    if not csv_files:
        raise RuntimeError("No CSVs were produced by FitCSVTool.")

    log_cb(f"Found {len(csv_files)} CSV file(s) from FitCSVTool.")

    main_csv = max(csv_files, key=lambda p: p.stat().st_size)
    df0 = pd.read_csv(main_csv)

    if not is_fitcsv_message_stream(df0):
        raise RuntimeError(
            "FitCSVTool output was not message-stream format, "
            "and the wide-table parsing path is not enabled in this version."
        )

    log_cb(f"Detected FitCSVTool message-stream CSV format: {main_csv.name}")

    max_i = _max_field_index(list(df0.columns))
    units_map: Dict[str, str] = {}
    enums_map: Dict[str, Dict[int, str]] = {}

    # First pass: get timestamps
    ts_list: List[float] = []
    for _, row in df0.iterrows():
        if str(row.get("Type", "")).strip() != "Data":
            continue
        pairs = _extract_pairs_from_row(row, max_i)
        if "timestamp" not in pairs:
            continue
        v, u = pairs["timestamp"]
        ts = _fit_timestamp_to_unix_seconds(v, u)
        if ts is not None:
            ts_list.append(ts)

    if not ts_list:
        raise RuntimeError("Message-stream CSV contains no timestamped Data rows (no 'timestamp' field found).")

    t0 = min(ts_list)
    t1 = max(ts_list)
    if t1 <= t0:
        raise RuntimeError("Timestamps exist but time range is zero/invalid.")

    # Second pass: collect all fields
    channel_raw: Dict[str, List[Tuple[float, Any]]] = defaultdict(list)

    for _, row in df0.iterrows():
        if str(row.get("Type", "")).strip() != "Data":
            continue
        msg = str(row.get("Message", "")).strip()
        if not msg:
            continue

        pairs = _extract_pairs_from_row(row, max_i)

        ts = None
        if "timestamp" in pairs:
            ts = _fit_timestamp_to_unix_seconds(*pairs["timestamp"])

        t_rel = 0.0 if ts is None else (ts - t0)

        for field_name, (val, unit) in pairs.items():
            chan = f"{msg}.{field_name}"

            vnum = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
            if pd.notna(vnum):
                channel_raw[chan].append((t_rel, float(vnum)))
            else:
                channel_raw[chan].append((t_rel, str(val)))

            if unit is not None and not (isinstance(unit, float) and math.isnan(unit)):
                units_map[chan] = str(unit)

    dtg = 1.0 / float(sample_hz)
    t_grid = np.arange(0.0, (t1 - t0) + dtg * 0.5, dtg, dtype=np.float64)

    out = {"Time": t_grid.astype(np.float32)}
    units_map.setdefault("Time", "s")

    for chan, tv in channel_raw.items():
        tv.sort(key=lambda x: x[0])
        t_raw = np.array([x[0] for x in tv], dtype=np.float64)
        y_raw_obj = [x[1] for x in tv]

        if all(isinstance(v, (int, float, np.floating)) for v in y_raw_obj):
            y_raw = np.array(y_raw_obj, dtype=np.float64)
            method = "linear" if resample_mode == "Linear (floats), ZOH (ints/strings)" else "zoh"
            out[chan] = resample_channel(t_raw, y_raw, t_grid, method).astype(np.float32)
        else:
            codes, uniques = pd.factorize(pd.Series(y_raw_obj, dtype="string").fillna(""), sort=False)
            enums_map[chan] = {int(i): str(u) for i, u in enumerate(list(uniques))}
            y_raw = codes.astype(np.float64)
            out[chan] = resample_channel(t_raw, y_raw, t_grid, "zoh").astype(np.float32)

    df_out = pd.DataFrame(out)
    return df_out, units_map, enums_map


# =============================================================================
# Worker + GUI
# =============================================================================

@dataclass
class ConvertConfig:
    fit_dir: Path
    fitcsvtool_jar: Path
    out_dir: Path
    sample_hz: float
    resample_mode: str
    meta: Dict[str, str]
    work_dir_base: Path


class ConvertWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cfg: ConvertConfig):
        super().__init__()
        self.cfg = cfg

    def _log(self, msg: str):
        self.progress.emit(msg)

    def run(self):
        t_all0 = time.time()
        try:
            fit_files = sorted([p for p in self.cfg.fit_dir.iterdir() if p.is_file() and p.suffix.lower() == ".fit"])
            if not fit_files:
                raise RuntimeError(f"No .fit files found in: {self.cfg.fit_dir}")

            self._log(f"Found {len(fit_files)} .fit file(s) in input folder.")

            n_skip = 0
            n_ok = 0
            n_fail = 0

            for idx, fit_path in enumerate(fit_files, start=1):
                out_ld_path = self.cfg.out_dir / (fit_path.stem + ".ld")

                if out_ld_path.exists():
                    n_skip += 1
                    self._log(f"[{idx}/{len(fit_files)}] SKIP: {fit_path.name} -> {out_ld_path.name} already exists")
                    continue

                # NEW: parse datetime from filename
                dt_header = parse_datetime_from_fit_filename(fit_path)
                if dt_header is not None:
                    self._log(f"[{idx}/{len(fit_files)}] CONVERT: {fit_path.name} (Date/Time={dt_header})")
                else:
                    self._log(f"[{idx}/{len(fit_files)}] CONVERT: {fit_path.name} (Date/Time=now)")

                work_dir = self.cfg.work_dir_base / f".fitcsv_{fit_path.stem}"
                work_dir.mkdir(parents=True, exist_ok=True)

                try:
                    df, units_map, enums_map = fit_to_dataframe_via_fitcsvtool(
                        fit_path=fit_path,
                        fitcsvtool_jar=self.cfg.fitcsvtool_jar,
                        temp_dir=work_dir,
                        sample_hz=self.cfg.sample_hz,
                        resample_mode=self.cfg.resample_mode,
                        log_cb=self._log,
                    )

                    self._log(f"Built dataframe with {len(df.columns)} channel(s). Writing LD...")

                    build_ld_from_dataframe(
                        df=df,
                        out_ld_path=str(out_ld_path),
                        sample_hz=self.cfg.sample_hz,
                        meta=self.cfg.meta,
                        units=units_map,
                        dt_for_header=dt_header,  # NEW
                    )

                    if enums_map:
                        sidecar = out_ld_path.with_suffix(".enums.json")
                        with open(sidecar, "w", encoding="utf-8") as f:
                            json.dump(enums_map, f, indent=2)
                        self._log(f"Wrote enum mapping sidecar: {sidecar.name}")

                    n_ok += 1
                    self._log(f"OK: {out_ld_path.name}")

                except Exception as e:
                    n_fail += 1
                    self._log(f"ERROR converting {fit_path.name}: {e}")

            dt_s = time.time() - t_all0
            self.finished.emit(
                f"Done in {dt_s:.2f}s. Converted: {n_ok}, Skipped: {n_skip}, Failed: {n_fail}."
            )

        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    SETTINGS_FILENAME = "fit_to_ld_gui_state.json"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FIT → MoTeC LD Converter (PyQt6)")

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # Paths group
        paths_box = QGroupBox("Paths")
        paths_layout = QFormLayout(paths_box)

        # Input folder
        self.fit_dir_edit = QLineEdit()
        self.fit_dir_browse_btn = QPushButton("Browse…")
        self.fit_dir_browse_btn.clicked.connect(self.browse_fit_dir)

        fit_row = QHBoxLayout()
        fit_row.addWidget(self.fit_dir_edit)
        fit_row.addWidget(self.fit_dir_browse_btn)
        paths_layout.addRow(QLabel("Input folder:"), self._wrap_row(fit_row))

        # Output folder
        self.out_dir_edit = QLineEdit()
        self.out_dir_browse_btn = QPushButton("Browse…")
        self.out_dir_browse_btn.clicked.connect(self.browse_out_dir)

        out_row = QHBoxLayout()
        out_row.addWidget(self.out_dir_edit)
        out_row.addWidget(self.out_dir_browse_btn)
        paths_layout.addRow(QLabel("Output folder:"), self._wrap_row(out_row))

        layout.addWidget(paths_box)

        # Options group
        opts_box = QGroupBox("Options")
        opts_layout = QFormLayout(opts_box)

        self.sample_hz_spin = QDoubleSpinBox()
        self.sample_hz_spin.setRange(0.5, 5000.0)
        self.sample_hz_spin.setDecimals(3)
        self.sample_hz_spin.setValue(50.0)
        opts_layout.addRow("Resample rate (Hz):", self.sample_hz_spin)

        self.resample_combo = QComboBox()
        self.resample_combo.addItems(
            [
                "Linear (floats), ZOH (ints/strings)",
                "ZOH for everything",
            ]
        )
        opts_layout.addRow("Resampling:", self.resample_combo)

        layout.addWidget(opts_box)

        # Metadata group
        meta_box = QGroupBox("MoTeC Metadata (optional)")
        meta_layout = QFormLayout(meta_box)

        self.driver_edit = QLineEdit()
        self.vehicle_edit = QLineEdit()
        self.venue_edit = QLineEdit()
        self.event_edit = QLineEdit()
        self.session_edit = QLineEdit()
        self.short_comment_edit = QLineEdit()

        meta_layout.addRow("Driver:", self.driver_edit)
        meta_layout.addRow("Vehicle ID:", self.vehicle_edit)
        meta_layout.addRow("Venue:", self.venue_edit)
        meta_layout.addRow("Event name:", self.event_edit)
        meta_layout.addRow("Event session:", self.session_edit)
        meta_layout.addRow("Short comment:", self.short_comment_edit)

        layout.addWidget(meta_box)

        # Controls
        controls = QHBoxLayout()
        self.convert_btn = QPushButton("Convert All")
        self.convert_btn.clicked.connect(self.start_convert)
        self.convert_btn.setDefault(True)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(lambda: self.log.setPlainText(""))

        controls.addWidget(self.convert_btn)
        controls.addWidget(self.clear_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        # Log
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self._thread: Optional[QThread] = None
        self._worker: Optional[ConvertWorker] = None

        # Persist UI state
        self._loading_state = False
        self.load_ui_state()
        self._connect_state_savers()

    def _settings_path(self) -> Path:
        return Path.cwd() / self.SETTINGS_FILENAME

    def _gather_ui_state(self) -> Dict[str, Any]:
        return {
            "fit_dir": self.fit_dir_edit.text().strip(),
            "out_dir": self.out_dir_edit.text().strip(),
            "sample_hz": float(self.sample_hz_spin.value()),
            "resample_mode": self.resample_combo.currentText(),
            "driver": self.driver_edit.text().strip(),
            "vehicleid": self.vehicle_edit.text().strip(),
            "venue": self.venue_edit.text().strip(),
            "event_name": self.event_edit.text().strip(),
            "event_session": self.session_edit.text().strip(),
            "short_comment": self.short_comment_edit.text().strip(),
        }

    def save_ui_state(self) -> None:
        if self._loading_state:
            return
        try:
            p = self._settings_path()
            state = self._gather_ui_state()
            with open(p, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.append_log(f"[state-save] warning: {e}")

    def load_ui_state(self) -> None:
        p = self._settings_path()
        if not p.is_file():
            return
        try:
            self._loading_state = True
            with open(p, "r", encoding="utf-8") as f:
                state = json.load(f)

            blockers = [
                QSignalBlocker(self.fit_dir_edit),
                QSignalBlocker(self.out_dir_edit),
                QSignalBlocker(self.sample_hz_spin),
                QSignalBlocker(self.resample_combo),
                QSignalBlocker(self.driver_edit),
                QSignalBlocker(self.vehicle_edit),
                QSignalBlocker(self.venue_edit),
                QSignalBlocker(self.event_edit),
                QSignalBlocker(self.session_edit),
                QSignalBlocker(self.short_comment_edit),
            ]
            _ = blockers

            self.fit_dir_edit.setText(state.get("fit_dir", "") or "")
            self.out_dir_edit.setText(state.get("out_dir", "") or "")

            if "sample_hz" in state:
                try:
                    self.sample_hz_spin.setValue(float(state["sample_hz"]))
                except Exception:
                    pass

            if "resample_mode" in state:
                mode = str(state["resample_mode"])
                idx = self.resample_combo.findText(mode)
                if idx >= 0:
                    self.resample_combo.setCurrentIndex(idx)

            self.driver_edit.setText(state.get("driver", "") or "")
            self.vehicle_edit.setText(state.get("vehicleid", "") or "")
            self.venue_edit.setText(state.get("venue", "") or "")
            self.event_edit.setText(state.get("event_name", "") or "")
            self.session_edit.setText(state.get("event_session", "") or "")
            self.short_comment_edit.setText(state.get("short_comment", "") or "")

        except Exception as e:
            self.append_log(f"[state-load] warning: {e}")
        finally:
            self._loading_state = False

    def _connect_state_savers(self) -> None:
        self.fit_dir_edit.textChanged.connect(lambda _: self.save_ui_state())
        self.out_dir_edit.textChanged.connect(lambda _: self.save_ui_state())
        self.sample_hz_spin.valueChanged.connect(lambda _: self.save_ui_state())
        self.resample_combo.currentTextChanged.connect(lambda _: self.save_ui_state())

        self.driver_edit.textChanged.connect(lambda _: self.save_ui_state())
        self.vehicle_edit.textChanged.connect(lambda _: self.save_ui_state())
        self.venue_edit.textChanged.connect(lambda _: self.save_ui_state())
        self.event_edit.textChanged.connect(lambda _: self.save_ui_state())
        self.session_edit.textChanged.connect(lambda _: self.save_ui_state())
        self.short_comment_edit.textChanged.connect(lambda _: self.save_ui_state())

    def _wrap_row(self, row: QHBoxLayout) -> QWidget:
        w = QWidget()
        w.setLayout(row)
        return w

    def append_log(self, msg: str):
        self.log.appendPlainText(msg)

    def browse_fit_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choose input folder (contains .fit files)", "")
        if path:
            self.fit_dir_edit.setText(path)
            self.save_ui_state()

    def browse_out_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choose output folder (for .ld files)", "")
        if path:
            self.out_dir_edit.setText(path)
            self.save_ui_state()

    def start_convert(self):
        fit_dir = Path(self.fit_dir_edit.text().strip())
        out_dir = Path(self.out_dir_edit.text().strip())
        jar_path = Path("FitCSVTool.jar")

        if not fit_dir.is_dir():
            QMessageBox.critical(self, "Error", "Input folder is invalid.")
            return
        if not out_dir.is_dir():
            QMessageBox.critical(self, "Error", "Output folder is invalid.")
            return
        if not jar_path.is_file():
            QMessageBox.critical(self, "Error", "FitCSVTool.jar not found in working directory.")
            return

        sample_hz = float(self.sample_hz_spin.value())
        resample_mode = self.resample_combo.currentText()

        meta = {
            "driver": self.driver_edit.text().strip(),
            "vehicleid": self.vehicle_edit.text().strip(),
            "venue": self.venue_edit.text().strip(),
            "event_name": self.event_edit.text().strip(),
            "event_session": self.session_edit.text().strip(),
            "short_comment": self.short_comment_edit.text().strip(),
            "vehicle_weight": "0",
            "vehicle_type": "",
            "vehicle_comment": "",
            "event_comment": "",
        }

        work_dir_base = out_dir / ".fitcsv_work"
        work_dir_base.mkdir(parents=True, exist_ok=True)

        self.save_ui_state()

        self.append_log("========================================")
        self.append_log(f"Input folder: {fit_dir}")
        self.append_log(f"Output folder: {out_dir}")
        self.append_log(f"Jar: {jar_path}")
        self.append_log(f"Work base: {work_dir_base}")
        self.append_log("Starting batch conversion…")

        self.convert_btn.setEnabled(False)

        cfg = ConvertConfig(
            fit_dir=fit_dir,
            fitcsvtool_jar=jar_path,
            out_dir=out_dir,
            sample_hz=sample_hz,
            resample_mode=resample_mode,
            meta=meta,
            work_dir_base=work_dir_base,
        )

        self._thread = QThread()
        self._worker = ConvertWorker(cfg)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.append_log)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)

        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_done(self, msg: str):
        self.append_log(msg)
        self.convert_btn.setEnabled(True)
        QMessageBox.information(self, "Done", msg)

    def _on_fail(self, err: str):
        self.append_log("ERROR:")
        self.append_log(err)
        self.convert_btn.setEnabled(True)
        QMessageBox.critical(self, "Conversion failed", err)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(900, 650)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
