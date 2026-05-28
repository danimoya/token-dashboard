#!/usr/bin/env python3
"""One-shot full bulk migrate from SQLite to HeliosDB. Run inside the
dashboard container as: python3 /sources/token-dashboard/scripts/_full_migrate.py"""
import os, sys, time, json
# HELIOSDB_DSN must be set in the environment (it is, in the dashboard
# container). Never hardcode the DSN/password here.
if not os.environ.get("HELIOSDB_DSN"):
    sys.exit("HELIOSDB_DSN not set — export it before running this migration")
# Prefer the host volume-mounted source so we can iterate on bulk_migrate.py
# without rebuilding the image.
sys.path.insert(0, "/app")
sys.path.insert(0, "/sources/token-dashboard")  # last insert = first in path
from token_dashboard import bulk_migrate
# Hot-reload bulk_migrate from /sources if we picked up the /app version
import importlib
if "/sources" not in (bulk_migrate.__file__ or ""):
    bulk_migrate = importlib.reload(bulk_migrate)
print(f"bulk_migrate from {bulk_migrate.__file__}", flush=True)

t0 = time.time()
print("starting migrate, limit=600000", flush=True)
r = bulk_migrate.migrate("/data/cache/token-dashboard.db", limit_messages=600000)
r["total_elapsed_s"] = round(time.time() - t0, 1)
with open("/tmp/migrate-result.json", "w") as f:
    json.dump(r, f, indent=2)
print(json.dumps(r, indent=2), flush=True)
