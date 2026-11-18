"""Reading the data contract: day directories written by marketrec."""

from .contract import SCHEMA_VERSION, TABLES, ContractError, Manifest, Run, read_manifest
from .day import Book20, Day
from .intervals import segment_ids, valid_intervals

__all__ = [
    "SCHEMA_VERSION",
    "TABLES",
    "Book20",
    "ContractError",
    "Day",
    "Manifest",
    "Run",
    "read_manifest",
    "segment_ids",
    "valid_intervals",
]
