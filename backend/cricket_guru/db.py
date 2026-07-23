"""Postgres connection helper."""
import psycopg2

from cricket_guru.config import PG


def connect():
    # search_path so bare table names resolve to cricsheet — the SQL model sometimes writes `matches`
    # instead of `cricsheet.matches`, and without this that errors and burns a whole retry round-trip.
    return psycopg2.connect(**PG, options="-c search_path=cricsheet,public")
