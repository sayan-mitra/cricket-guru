"""Postgres connection helper."""
import psycopg2

from cricket_guru.config import PG


def connect():
    return psycopg2.connect(**PG)
