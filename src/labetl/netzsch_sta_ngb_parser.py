"""
NETZSCH STA NGB File Parser
"""

from __future__ import annotations

import logging
import re
import struct
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from itertools import tee, zip_longest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from polars.exceptions import ShapeError

from labetl.util import get_hash, set_metadata

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
# Avoid "No handler found" warnings for library users.
logger.addHandler(logging.NullHandler())


# -----------------------------------------------------------------------------
# Constants and Configuration
# -----------------------------------------------------------------------------
class DataType(Enum):
    """Binary data type identifiers."""

    INT32 = b"\x03"
    FLOAT32 = b"\x04"
    FLOAT64 = b"\x05"
    STRING = b"\x1f"


@dataclass(frozen=True)
class BinaryMarkers:
    """Binary markers for parsing NGB files."""

    END_FIELD: bytes = rb"\x01\x00\x00\x00\x02\x00\x01\x00\x00"
    TYPE_PREFIX: bytes = rb"\x17\xfc\xff\xff"
    TYPE_SEPARATOR: bytes = rb"\x80\x01"
    END_TABLE: bytes = rb"\x18\xfc\xff\xff\x03"
    TABLE_SEPARATOR: bytes = rb"\x00\x00\x01\x00\x00\x00\x0c\x00\x17\xfc\xff\xff\x1a\x80\x01\x01\x80\x02\x00\x00"
    START_DATA: bytes = b"\xa0\x01"
    END_DATA: bytes = (
        b"\x01\x00\x00\x00\x02\x00\x01\x00\x00\x00\x03\x00\x18\xfc\xff\xff\x03\x80\x01"
    )


@dataclass
class PatternConfig:
    """Configuration for metadata and column patterns."""

    metadata_patterns: Dict[str, Tuple[bytes, bytes]] = field(
        default_factory=lambda: {
            "instrument": (rb"\x75\x17", rb"\x59\x10"),
            "project": (rb"\x72\x17", rb"\x3c\x08"),
            "date_performed": (rb"\x72\x17", rb"\x3e\x08"),
            "lab": (rb"\x72\x17", rb"\x34\x08"),
            "operator": (rb"\x72\x17", rb"\x35\x08"),
            "crucible_type": (rb"\x7e\x17", rb"\x40\x08"),
            "comment": (rb"\x72\x17", rb"\x3d\x08"),
            "furnace_type": (rb"\x7a\x17", rb"\x40\x08"),
            "carrier_type": (rb"\x79\x17", rb"\x40\x08"),
            "sample_id": (rb"\x30\x75", rb"\x98\x08"),
            "sample_name": (rb"\x30\x75", rb"\x40\x08"),
            "sample_mass": (rb"\x30\x75", rb"\x9e\x0c"),
            "crucible_mass": (rb"\x7e\x17", rb"\x9e\x0c"),
            "material": (rb"\x30\x75", rb"\x62\x09"),
        }
    )
    temp_prog_patterns: Dict[str, bytes] = field(
        default_factory=lambda: {
            "stage_type": rb"\x3f\x08",
            "temperature": rb"\x17\x0e",
            "heating_rate": rb"\x13\x0e",
            "acquisition_rate": rb"\x14\x0e",
            "time": rb"\x15\x0e",
        }
    )
    cal_constants_patterns: Dict[str, bytes] = field(
        default_factory=lambda: {
            f"p{i}": bytes([0x4F + i, 0x04]) if i < 5 else b"\xc3\x04" for i in range(6)
        }
    )
    column_map: Dict[str, str] = field(
        default_factory=lambda: {
            "8d": "time",
            "8e": "temperature",
            "9c": "dsc",
            "9e": "purge_flow",
            "90": "protective_flow",
            "87": "sample_mass",
            "30": "furnace_temperature",
            "32": "furnace_power",
            "33": "h_foil_temp",
            "34": "uc_module",
            "35": "env_pressure",
            "36": "env_accel_x",
            "37": "env_accel_y",
            "38": "env_accel_z",
        }
    )


# -----------------------------------------------------------------------------
# Parser Primitives
# -----------------------------------------------------------------------------
class BinaryParser:
    """Handles binary data parsing operations."""

    def __init__(self, markers: Optional[BinaryMarkers] = None):
        self.markers = markers or BinaryMarkers()
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        # Hot-path: table separator compiled once
        self._compiled_patterns["table_sep"] = re.compile(
            self.markers.TABLE_SEPARATOR, re.DOTALL
        )

    def _get_compiled_pattern(self, key: str, pattern: bytes) -> re.Pattern:
        """Cache compiled regex patterns for performance."""
        pat = self._compiled_patterns.get(key)
        if pat is None:
            pat = re.compile(pattern, re.DOTALL)
            self._compiled_patterns[key] = pat
        return pat

    @staticmethod
    def parse_value(data_type: bytes, value: bytes) -> Any:
        """Parse binary value based on data type."""
        try:
            if data_type == DataType.INT32.value:
                return struct.unpack("<i", value)[0]
            if data_type == DataType.FLOAT32.value:
                return struct.unpack("<f", value)[0]
            if data_type == DataType.FLOAT64.value:
                return struct.unpack("<d", value)[0]
            if data_type == DataType.STRING.value:
                # Skip 4-byte length; strip nulls.
                return (
                    value[4:]
                    .decode("utf-8", errors="ignore")
                    .strip()
                    .replace("\x00", "")
                )
            return value
        except Exception as e:
            logger.debug("Failed to parse value: %s", e)
            return None

    def split_tables(self, data: bytes) -> List[bytes]:
        """Split binary data into tables using the known separator."""
        pattern = self._compiled_patterns["table_sep"]
        indices = [m.start() - 2 for m in pattern.finditer(data)]
        if not indices:
            return [data]
        start, end = tee(indices)
        next(end, None)
        return [data[i:j] for i, j in zip_longest(start, end)]

    def extract_data_array(self, table: bytes, data_type: bytes) -> List[float]:
        """Extract array of numerical data from a table (float32/64)."""
        mv = memoryview(table)
        start_idx = mv.tobytes().find(self.markers.START_DATA)
        if start_idx == -1:
            return []
        start_idx += 6  # preserve original offset logic
        data_mv = mv[start_idx:]
        end_idx = data_mv.tobytes().find(self.markers.END_DATA)
        if end_idx == -1:
            return []
        chunk = data_mv[:end_idx].tobytes()

        if data_type == DataType.FLOAT64.value:
            return [x[0] for x in struct.iter_unpack("<d", chunk)]
        if data_type == DataType.FLOAT32.value:
            return [x[0] for x in struct.iter_unpack("<f", chunk)]
        return []


class MetadataExtractor:
    """Extracts metadata from NGB tables."""

    def __init__(self, config: PatternConfig, parser: BinaryParser):
        self.config = config
        self.parser = parser
        self._compiled_meta: Dict[str, re.Pattern] = {}
        self._compiled_temp_prog: Dict[str, re.Pattern] = {}
        self._compiled_cal_consts: Dict[str, re.Pattern] = {}

        # Precompile patterns used in tight loops for speed (logic unchanged).
        END_FIELD = self.parser.markers.END_FIELD
        TYPE_PREFIX = self.parser.markers.TYPE_PREFIX
        TYPE_SEPARATOR = self.parser.markers.TYPE_SEPARATOR

        for fname, (category, field) in self.config.metadata_patterns.items():
            pat = (
                category
                + rb".+?"
                + field
                + rb".+?"
                + TYPE_PREFIX
                + rb"(.+?)"
                + TYPE_SEPARATOR
                + rb"(.+?)"
                + END_FIELD
            )
            self._compiled_meta[fname] = re.compile(pat, re.DOTALL)

        for fname, pat_bytes in self.config.temp_prog_patterns.items():
            pat = (
                pat_bytes
                + rb".+?"
                + TYPE_PREFIX
                + rb"(.+?)"
                + TYPE_SEPARATOR
                + rb"(.+?)"
                + END_FIELD
            )
            self._compiled_temp_prog[fname] = re.compile(pat, re.DOTALL)

        for fname, pat_bytes in self.config.cal_constants_patterns.items():
            pat = (
                pat_bytes
                + rb".+?"
                + TYPE_PREFIX
                + rb"(.+?)"
                + TYPE_SEPARATOR
                + rb"(.+?)"
                + END_FIELD
            )
            self._compiled_cal_consts[fname] = re.compile(pat, re.DOTALL)

    def extract_field(self, table: bytes, field_name: str) -> Optional[Any]:
        """Extract a single metadata field (value only)."""
        pattern = self._compiled_meta[field_name]
        matches = pattern.findall(table)
        if matches:
            data_type, value = matches[0]
            return self.parser.parse_value(data_type, value)
        return None

    def extract_metadata(self, tables: List[bytes]) -> Dict[str, Any]:
        """Extract all metadata from tables."""
        metadata: Dict[str, Any] = {}

        for table in tables:
            # Standard fields
            for field_name in self._compiled_meta.keys():
                value = self.extract_field(table, field_name)
                if value is not None:
                    if field_name == "date_performed" and isinstance(value, int):
                        value = datetime.fromtimestamp(
                            value, tz=timezone.utc
                        ).isoformat()
                    metadata[field_name] = value

            # Temperature program
            self._extract_temperature_program(table, metadata)

            # Calibration constants
            self._extract_calibration_constants(table, metadata)

        return metadata

    def _extract_temperature_program(
        self, table: bytes, metadata: Dict[str, Any]
    ) -> None:
        """Extract temperature program section."""
        CATEGORY = b"\x0c\x2b"
        if CATEGORY not in table:
            return

        step_num = table[0:2].decode("ascii", errors="ignore")[0] if table else "0"
        temp_prog = metadata.setdefault("temperature_program", {})
        step_key = f"step_{step_num}"
        step_data = temp_prog.setdefault(step_key, {})

        for field_name, pattern in self._compiled_temp_prog.items():
            match = pattern.search(table)
            if match:
                data_type, value_bytes = match.groups()
                value = self.parser.parse_value(data_type, value_bytes)
                if value is not None:
                    step_data[field_name] = value

    def _extract_calibration_constants(
        self, table: bytes, metadata: Dict[str, Any]
    ) -> None:
        """Extract calibration constants section."""
        CATEGORY = b"\xf5\x01"
        if CATEGORY not in table:
            return

        cal_constants = metadata.setdefault("calibration_constants", {})
        for field_name, pattern in self._compiled_cal_consts.items():
            match = pattern.search(table)
            if match:
                data_type, value_bytes = match.groups()
                value = self.parser.parse_value(data_type, value_bytes)
                if value is not None:
                    cal_constants[field_name] = value


class DataStreamProcessor:
    """Processes data streams from NGB files."""

    def __init__(self, config: PatternConfig, parser: BinaryParser):
        self.config = config
        self.parser = parser
        self._table_sep_re = self.parser._get_compiled_pattern(
            "table_sep", self.parser.markers.TABLE_SEPARATOR
        )

    # --- Stream 2 ---
    def process_stream_2(self, stream_data: bytes) -> pl.DataFrame:
        """Process primary data stream (stream_2)."""
        # Split into tables - exact original logic
        indices = [m.start() - 2 for m in self._table_sep_re.finditer(stream_data)]
        start, end = tee(indices)
        next(end, None)
        stream_table = [stream_data[i:j] for i, j in zip_longest(start, end)]

        output: List[float] = []
        output_polars = pl.DataFrame()
        title: Optional[str] = None

        col_map = self.config.column_map
        markers = self.parser.markers

        for table in stream_table:
            if table[1:2] == b"\x17":  # header
                title = table[0:1].hex()
                title = col_map.get(title, title)
                if len(output) > 1:
                    try:
                        output_polars = output_polars.with_columns(
                            pl.Series(name=title, values=output)
                        )
                    except ShapeError:
                        logger.debug("Shape mismatch when adding column '%s'", title)
                output = []

            if table[1:2] == b"\x75":  # data
                start_data = table.find(markers.START_DATA) + 6
                data = table[start_data:]
                end_data = data.find(markers.END_DATA)
                data = data[:end_data]
                data_type = table[start_data - 7 : start_data - 6]

                if data_type == DataType.FLOAT64.value:
                    output.extend(x[0] for x in struct.iter_unpack("<d", data))
                elif data_type == DataType.FLOAT32.value:
                    output.extend(x[0] for x in struct.iter_unpack("<f", data))

        return output_polars

    # --- Stream 3 ---
    def process_stream_3(
        self, stream_data: bytes, existing_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Process secondary data stream (stream_3)."""
        # Split into tables - exact original logic
        indices = [m.start() - 2 for m in self._table_sep_re.finditer(stream_data)]
        start, end = tee(indices)
        next(end, None)
        stream_table = [stream_data[i:j] for i, j in zip_longest(start, end)]

        output: List[float] = []
        output_polars = existing_df
        title: Optional[str] = None

        col_map = self.config.column_map
        markers = self.parser.markers

        for table in stream_table:
            if table[22:25] == b"\x80\x22\x2b":  # header
                title = table[0:1].hex()
                title = col_map.get(title, title)
                output = []

            if table[1:2] == b"\x75":  # data
                start_data = table.find(markers.START_DATA) + 6
                data = table[start_data:]
                end_data = data.find(markers.END_DATA)
                data = data[:end_data]
                data_type = table[start_data - 7 : start_data - 6]

                if data_type == DataType.FLOAT64.value:
                    output.extend(x[0] for x in struct.iter_unpack("<d", data))
                elif data_type == DataType.FLOAT32.value:
                    output.extend(x[0] for x in struct.iter_unpack("<f", data))

                # Save after each data block (original behavior)
                try:
                    output_polars = output_polars.with_columns(
                        pl.Series(name=title, values=output)
                    )
                except ShapeError:
                    # Silently ignore shape issues as before
                    pass

        return output_polars


# -----------------------------------------------------------------------------
# Main Parser
# -----------------------------------------------------------------------------
class NGBParser:
    """Main parser for NETZSCH STA NGB files."""

    def __init__(self, config: Optional[PatternConfig] = None):
        self.config = config or PatternConfig()
        self.markers = BinaryMarkers()
        self.binary_parser = BinaryParser(self.markers)
        self.metadata_extractor = MetadataExtractor(self.config, self.binary_parser)
        self.data_processor = DataStreamProcessor(self.config, self.binary_parser)

    def parse(self, path: str) -> Tuple[Dict[str, Any], pa.Table]:
        """Parse NGB file and return (metadata, Arrow table)."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")

        metadata: Dict[str, Any] = {}
        data_df = pl.DataFrame()

        try:
            with zipfile.ZipFile(path, "r") as z:
                # stream_1: metadata
                if "Streams/stream_1.table" in z.namelist():
                    with z.open("Streams/stream_1.table") as stream:
                        stream_data = stream.read()
                        tables = self.binary_parser.split_tables(stream_data)
                        metadata = self.metadata_extractor.extract_metadata(tables)

                # stream_2: primary data
                if "Streams/stream_2.table" in z.namelist():
                    with z.open("Streams/stream_2.table") as stream:
                        stream_data = stream.read()
                        data_df = self.data_processor.process_stream_2(stream_data)

                # stream_3: additional data merged into existing df
                if "Streams/stream_3.table" in z.namelist():
                    with z.open("Streams/stream_3.table") as stream:
                        stream_data = stream.read()
                        data_df = self.data_processor.process_stream_3(
                            stream_data, data_df
                        )

        except Exception as e:
            logger.error("Failed to parse NGB file: %s", e)
            raise

        return metadata, data_df.to_arrow()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def load_ngb_data(path: str) -> pa.Table:
    """
    Load a STA file and store metadata in the PyArrow table.

    Parameters
    ----------
    path : str
        The path to the STA file.

    Returns
    -------
    pa.Table
        Table with data and embedded metadata.
    """
    parser = NGBParser()
    metadata, data = parser.parse(path)

    # Add file hash to metadata
    file_hash = get_hash(path)
    metadata["file_hash"] = {
        "file": Path(path).name,
        "method": "BLAKE2b",
        "hash": file_hash,
    }

    # Attach metadata to the Arrow table
    data = set_metadata(data, tbl_meta={"file_metadata": metadata, "type": "STA"})
    return data


def get_sta_data(path: str) -> Tuple[Dict[str, Any], pa.Table]:
    """
    Get STA data and metadata from an NGB file.

    Parameters
    ----------
    path : str
        Path to the .ngb-ss3 file.

    Returns
    -------
    (dict, pa.Table)
        Tuple of (metadata dict, Arrow table).
    """
    parser = NGBParser()
    return parser.parse(path)


# -----------------------------------------------------------------------------
# Extended functionality for future development
# -----------------------------------------------------------------------------
class NGBParserExtended(NGBParser):
    """Extended parser with additional capabilities."""

    def __init__(
        self, config: Optional[PatternConfig] = None, cache_patterns: bool = True
    ):
        super().__init__(config)
        self.cache_patterns = cache_patterns
        self._pattern_cache: Dict[str, re.Pattern] = {}

    def add_custom_column_mapping(self, hex_id: str, column_name: str) -> None:
        """Add custom column mapping at runtime."""
        self.config.column_map[hex_id] = column_name

    def add_metadata_pattern(
        self, field_name: str, category: bytes, field: bytes
    ) -> None:
        """Add custom metadata pattern at runtime."""
        self.config.metadata_patterns[field_name] = (category, field)

    def parse_with_validation(self, path: str) -> Tuple[Dict[str, Any], pa.Table]:
        """Parse with additional validation."""
        metadata, data = self.parse(path)

        # Validate required columns
        required_columns = ["time", "temperature"]
        schema = data.schema
        missing = [col for col in required_columns if col not in schema.names]
        if missing:
            logger.warning("Missing required columns: %s", missing)

        # Validate data ranges
        if "temperature" in schema.names:
            temp_col = data.column("temperature").to_pylist()
            if temp_col and (min(temp_col) < -273.15 or max(temp_col) > 3000):
                logger.warning("Temperature values outside expected range")

        return metadata, data


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> int:
    """Command-line interface for the parser."""
    import argparse

    parser_cli = argparse.ArgumentParser(description="Parse NETZSCH STA NGB files")
    parser_cli.add_argument("input", help="Input NGB file path")
    parser_cli.add_argument("-o", "--output", help="Output directory", default=".")
    parser_cli.add_argument(
        "-f",
        "--format",
        choices=["parquet", "csv", "all"],
        default="parquet",
        help="Output format",
    )
    parser_cli.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser_cli.parse_args()

    logging.basicConfig(level=(logging.DEBUG if args.verbose else logging.INFO))

    try:
        data = load_ngb_data(args.input)
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        base_name = Path(args.input).stem
        if args.format in ("parquet", "all"):
            pq.write_table(
                data, output_path / f"{base_name}.parquet", compression="snappy"
            )
        if args.format in ("csv", "all"):
            df = pl.from_arrow(data).to_pandas()
            df.to_csv(output_path / f"{base_name}.csv", index=False)

        logger.info("Successfully parsed %s", args.input)
        return 0
    except Exception as e:
        logger.error("Failed to parse file: %s", e)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
