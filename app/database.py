from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

# Vercel only allows writes to /tmp/; local uses data/
_VERCEL = os.environ.get("VERCEL", "") == "1" or bool(os.environ.get("VERCEL_ENV", ""))
if _VERCEL:
    DB_PATH = Path("/tmp/kdetect.db")
else:
    DB_PATH = Path(__file__).resolve().parents[1] / "data" / "kdetect.db"


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                quality_score INTEGER NOT NULL,
                defects TEXT NOT NULL,
                metrics TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_prediction(filename: str, result: dict[str, Any]) -> int:
    init_db()
    with sqlite3.connect(str(DB_PATH)) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
            (filename, prediction, confidence, quality_score, defects, metrics)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                result["quality"],
                result["confidence"],
                result["quality_score"],
                json.dumps(result["detected_defects"]),
                json.dumps(result["metrics"]),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_predictions(limit: int = 25) -> list[dict[str, Any]]:
    init_db()
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "filename": row["filename"],
            "prediction": row["prediction"],
            "confidence": row["confidence"],
            "quality_score": row["quality_score"],
            "defects": json.loads(row["defects"]),
            "metrics": json.loads(row["metrics"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]
