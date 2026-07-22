#!/usr/bin/env bash
# All-in-one boot: Postgres (on the persistent volume) + Streamlit (with embedded on-disk Qdrant).
# First boot initialises Postgres and restores the baked Cricsheet dump; later boots just start it.
set -e

export PGDATA=${PGDATA:-/data/pgdata}
PGBIN=$(ls -d /usr/lib/postgresql/*/bin | head -1)
export PATH="$PGBIN:$PATH"

# Unpack the baked vector index into place (embedded on-disk Qdrant reads data/qdrant).
[ -d /app/data/qdrant ] || tar xzf /app/data/qdrant.tar.gz -C /app/data

mkdir -p "$PGDATA"
chown -R postgres:postgres "$(dirname "$PGDATA")"

if [ ! -s "$PGDATA/PG_VERSION" ]; then
  echo ">> first boot: initialising Postgres"
  gosu postgres initdb -D "$PGDATA" >/dev/null
  echo "host all all 127.0.0.1/32 trust" >> "$PGDATA/pg_hba.conf"
  echo "listen_addresses='127.0.0.1'"    >> "$PGDATA/postgresql.conf"
  gosu postgres pg_ctl -D "$PGDATA" -w start
  gosu postgres createuser -s cricket
  gosu postgres createdb -O cricket cricket_guru
  echo ">> restoring Cricsheet dump (~35MB gz)"
  gunzip -c /app/deploy/cricket_guru.dump.gz | gosu postgres psql -q -d cricket_guru
  gosu postgres pg_ctl -D "$PGDATA" -w stop
  echo ">> Postgres ready"
fi

gosu postgres pg_ctl -D "$PGDATA" -w start

echo ">> starting Streamlit on port ${PORT:-8501}"
exec streamlit run frontend/app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"
