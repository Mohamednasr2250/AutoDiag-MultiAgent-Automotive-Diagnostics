import sqlite3, json, os
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

DB_PATH = os.environ.get("DB_PATH", "autodiag.db")

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn; conn.commit()
    except Exception:
        conn.rollback(); raise
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT DEFAULT '',
            query TEXT NOT NULL, answer TEXT NOT NULL, rating BOOLEAN NOT NULL,
            comment TEXT DEFAULT '', timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS episodic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            symptoms TEXT NOT NULL, diagnosis TEXT NOT NULL, outcome TEXT DEFAULT '',
            access_count INTEGER DEFAULT 0, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS semantic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL, confidence REAL DEFAULT 0.8, source TEXT DEFAULT 'agent',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS procedural_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT, trigger TEXT UNIQUE NOT NULL,
            tool_sequence TEXT NOT NULL, success_rate REAL DEFAULT 1.0,
            use_count INTEGER DEFAULT 0, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS consolidation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, episodes_before INTEGER,
            episodes_after INTEGER, removed INTEGER, ran_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_session ON episodic_memory(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_key ON semantic_memory(key)")

def insert_feedback(query, answer, rating, comment="", session_id=""):
    with get_conn() as conn:
        cursor = conn.execute("INSERT INTO feedback (session_id,query,answer,rating,comment) VALUES (?,?,?,?,?)",
            (session_id, query, answer[:300], rating, comment))
        row_id = cursor.lastrowid
    return get_feedback_by_id(row_id)

def get_feedback_by_id(fid):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM feedback WHERE id=?", (fid,)).fetchone()
    return dict(row) if row else None

def get_all_feedback():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM feedback ORDER BY timestamp DESC").fetchall()
    return [dict(r) for r in rows]

def get_feedback_by_session(session_id):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM feedback WHERE session_id=? ORDER BY timestamp DESC", (session_id,)).fetchall()
    return [dict(r) for r in rows]

def get_feedback_stats():
    with get_conn() as conn:
        stats = conn.execute("""SELECT COUNT(*) as total,
            SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN rating=0 THEN 1 ELSE 0 END) as negative,
            ROUND(AVG(CASE WHEN rating=1 THEN 100.0 ELSE 0 END),1) as satisfaction_rate
            FROM feedback""").fetchone()
        recent_neg = conn.execute("SELECT * FROM feedback WHERE rating=0 ORDER BY timestamp DESC LIMIT 5").fetchall()
    if not stats or stats["total"] == 0:
        return {"total":0,"positive":0,"negative":0,"satisfaction_rate":0,"recent_negative":[]}
    return {"total":stats["total"],"positive":stats["positive"] or 0,"negative":stats["negative"] or 0,
            "satisfaction_rate":stats["satisfaction_rate"] or 0,"recent_negative":[dict(r) for r in recent_neg]}

def insert_episode(session_id, symptoms, diagnosis, outcome=""):
    with get_conn() as conn:
        cursor = conn.execute("INSERT INTO episodic_memory (session_id,symptoms,diagnosis,outcome) VALUES (?,?,?,?)",
            (session_id, symptoms[:500], diagnosis[:1000], outcome))
        return cursor.lastrowid

def get_similar_episodes(symptoms, k=3):
    words = symptoms.lower().split()[:10]
    with get_conn() as conn:
        all_eps = conn.execute("SELECT * FROM episodic_memory ORDER BY access_count DESC").fetchall()
        scored = [(sum(1 for w in words if w in ep["symptoms"].lower()), dict(ep)) for ep in all_eps]
        scored = [(s,e) for s,e in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [e for _,e in scored[:k]]
        for ep in results:
            conn.execute("UPDATE episodic_memory SET access_count=access_count+1,last_accessed=CURRENT_TIMESTAMP WHERE id=?", (ep["id"],))
    return results

def get_all_episodes():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM episodic_memory ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]

def delete_duplicate_episodes():
    with get_conn() as conn:
        before = conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        conn.execute("""DELETE FROM episodic_memory WHERE id NOT IN (
            SELECT MIN(id) FROM episodic_memory GROUP BY SUBSTR(symptoms,1,50),SUBSTR(diagnosis,1,50))""")
        after = conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
    return before - after

def decay_old_episodes(days=30):
    with get_conn() as conn:
        before = conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        conn.execute("DELETE FROM episodic_memory WHERE created_at < datetime('now', ? || ' days') AND access_count < 5", (f"-{days}",))
        after = conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
    return before - after

def upsert_fact(key, value, confidence=0.8, source="agent"):
    if confidence < 0.5: return False
    with get_conn() as conn:
        conn.execute("""INSERT INTO semantic_memory (key,value,confidence,source) VALUES (?,?,?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value,confidence=excluded.confidence,source=excluded.source""",
            (key, value, confidence, source))
    return True

def get_fact(key):
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM semantic_memory WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None

def get_all_facts():
    with get_conn() as conn:
        rows = conn.execute("SELECT key,value FROM semantic_memory").fetchall()
    return {r["key"]: r["value"] for r in rows}

def upsert_procedure(trigger, tool_sequence, success_rate=1.0):
    if success_rate < 0.7: return False
    with get_conn() as conn:
        conn.execute("""INSERT INTO procedural_memory (trigger,tool_sequence,success_rate) VALUES (?,?,?)
            ON CONFLICT(trigger) DO UPDATE SET tool_sequence=excluded.tool_sequence,success_rate=excluded.success_rate""",
            (trigger, json.dumps(tool_sequence), success_rate))
    return True

def get_procedure(trigger):
    with get_conn() as conn:
        row = conn.execute("SELECT tool_sequence FROM procedural_memory WHERE trigger=?", (trigger,)).fetchone()
        if row: conn.execute("UPDATE procedural_memory SET use_count=use_count+1 WHERE trigger=?", (trigger,))
    return json.loads(row["tool_sequence"]) if row else None

def get_best_procedure(fault_codes):
    with get_conn() as conn:
        rows = conn.execute("SELECT trigger,tool_sequence FROM procedural_memory ORDER BY success_rate DESC").fetchall()
    for row in rows:
        if any(code in fault_codes for code in row["trigger"].split(",")): return json.loads(row["tool_sequence"])
    return None

def log_consolidation(before, after, removed):
    with get_conn() as conn:
        conn.execute("INSERT INTO consolidation_log (episodes_before,episodes_after,removed) VALUES (?,?,?)", (before,after,removed))

def get_db_stats():
    with get_conn() as conn:
        return {
            "feedback_count": conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0],
            "episodic_count": conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0],
            "semantic_count": conn.execute("SELECT COUNT(*) FROM semantic_memory").fetchone()[0],
            "procedural_count": conn.execute("SELECT COUNT(*) FROM procedural_memory").fetchone()[0],
            "db_path": DB_PATH}

init_db()