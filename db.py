"""
Read-only Mongo accessor used by the FastAPI insights service.

We open one client at module import and reuse it across requests, the
same pattern Spring Data uses on the Java side. The connection string
comes from the {MONGODB_URI} environment variable, defaulting to the
same local Mongo Spring talks to.
"""
from __future__ import annotations

import os
from functools import lru_cache

from pymongo import MongoClient
from pymongo.database import Database


@lru_cache(maxsize=1)
def get_db() -> Database:
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/workflow_engine")
    # `MongoClient` parses the database name out of the URI when present.
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    db_name = uri.rsplit("/", 1)[-1].split("?", 1)[0] or "workflow_engine"
    return client[db_name]
