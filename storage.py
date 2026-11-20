"""The crate: every analyzed track, with the user's edits.

Two interchangeable backends with the same methods:
  - FirestoreStore: used on Cloud Run (persistent, survives container restarts)
  - LocalStore: a JSON file, for running the container on your own machine
    without Google credentials (wiped when the container is removed)

Pick with the CRATE_STORE env var ("firestore" or "local"). By default Cloud Run
(which sets K_SERVICE) uses Firestore and everything else uses the local file.
"""
import json
import os
import threading
import time
import uuid

COLLECTION = "tracks"


def make_store():
    kind = os.getenv("CRATE_STORE", "firestore" if os.getenv("K_SERVICE") else "local")
    return FirestoreStore() if kind == "firestore" else LocalStore()


class LocalStore:
    kind = "local"

    def __init__(self, path: str = ""):
        self.path = path or os.getenv("CRATE_PATH", "/tmp/crate.json")
        path = self.path
        self.lock = threading.Lock()
        if not os.path.exists(path):
            self._save({})

    def _load(self) -> dict:
        with open(self.path) as f:
            return json.load(f)

    def _save(self, data: dict):
        with open(self.path, "w") as f:
            json.dump(data, f)

    def add(self, record: dict) -> str:
        rid = uuid.uuid4().hex[:12]
        with self.lock:
            data = self._load()
            data[rid] = {**record, "id": rid}
            self._save(data)
        return rid

    def list(self, limit: int = 60) -> list:
        with self.lock:
            recs = list(self._load().values())
        return sorted(recs, key=lambda r: r["created_at"], reverse=True)[:limit]

    def get(self, rid: str):
        with self.lock:
            return self._load().get(rid)

    def update_edits(self, rid: str, edits: dict) -> bool:
        with self.lock:
            data = self._load()
            if rid not in data:
                return False
            data[rid]["edits"] = edits
            data[rid]["updated_at"] = time.time()
            self._save(data)
        return True

    def delete(self, rid: str) -> bool:
        with self.lock:
            data = self._load()
            existed = data.pop(rid, None) is not None
            self._save(data)
        return existed


class FirestoreStore:
    kind = "firestore"

    def __init__(self):
        from google.cloud import firestore   # imported here so local runs don't need credentials
        self.db = firestore.Client()
        self.col = self.db.collection(COLLECTION)
        self._desc = firestore.Query.DESCENDING

    def add(self, record: dict) -> str:
        ref = self.col.document()
        ref.set({**record, "id": ref.id})
        return ref.id

    def list(self, limit: int = 60) -> list:
        q = self.col.order_by("created_at", direction=self._desc).limit(limit)
        return [d.to_dict() for d in q.stream()]

    def get(self, rid: str):
        snap = self.col.document(rid).get()
        return snap.to_dict() if snap.exists else None

    def update_edits(self, rid: str, edits: dict) -> bool:
        ref = self.col.document(rid)
        if not ref.get().exists:
            return False
        ref.update({"edits": edits, "updated_at": time.time()})
        return True

    def delete(self, rid: str) -> bool:
        ref = self.col.document(rid)
        if not ref.get().exists:
            return False
        ref.delete()
        return True
