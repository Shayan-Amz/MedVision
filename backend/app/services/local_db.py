import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "medvision_local.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS studies_history (
            id TEXT PRIMARY KEY, filename TEXT, patient_info TEXT,
            primary_finding TEXT, confidence REAL, urgency TEXT,
            clinical_impression TEXT, created_at TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS dpo_pairs (
            id TEXT PRIMARY KEY, study_id TEXT, filename TEXT,
            finding TEXT, rejected TEXT, chosen TEXT, created_at TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS n8n_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, workflow TEXT,
            event TEXT, status TEXT, created_at TEXT)""")
        conn.commit()

def save_dpo_pair(pair_id, study_id, filename, finding, rejected, chosen, ts):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR REPLACE INTO dpo_pairs VALUES (?,?,?,?,?,?,?)",
                     (pair_id, study_id, filename, finding, rejected, chosen, ts))
        conn.commit()

def query_dpo_pairs(q: str = "", page: int = 1, page_size: int = 10):
    offset = max(0, (page - 1) * page_size)
    like = f"%{q.strip()}%"
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if q.strip():
            total = conn.execute("SELECT COUNT(*) FROM dpo_pairs WHERE finding LIKE ? OR chosen LIKE ? OR rejected LIKE ?", (like, like, like)).fetchone()[0]
            rows = conn.execute("SELECT * FROM dpo_pairs WHERE finding LIKE ? OR chosen LIKE ? OR rejected LIKE ? ORDER BY created_at DESC LIMIT ? OFFSET ?", (like, like, like, page_size, offset)).fetchall()
        else:
            total = conn.execute("SELECT COUNT(*) FROM dpo_pairs").fetchone()[0]
            rows = conn.execute("SELECT * FROM dpo_pairs ORDER BY created_at DESC LIMIT ? OFFSET ?", (page_size, offset)).fetchall()
        return [dict(r) for r in rows], total

def delete_dpo_pair(pair_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM dpo_pairs WHERE id = ?", (pair_id,))
        conn.commit()

def save_n8n_log(workflow, event, status, ts):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO n8n_logs (workflow, event, status, created_at) VALUES (?,?,?,?)",
                     (workflow, event, status, ts))
        conn.commit()

def query_n8n_logs(q: str = "", status_filter: str = "", page: int = 1, page_size: int = 10):
    offset = max(0, (page - 1) * page_size)
    clauses, params = [], []
    if q.strip():
        clauses.append("(event LIKE ? OR workflow LIKE ?)")
        params.extend([f"%{q.strip()}%", f"%{q.strip()}%"])
    if status_filter.strip():
        clauses.append("status LIKE ?")
        params.append(f"%{status_filter.strip()}%")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute(f"SELECT COUNT(*) FROM n8n_logs {where}", params).fetchone()[0]
        rows = conn.execute(f"SELECT * FROM n8n_logs {where} ORDER BY id DESC LIMIT ? OFFSET ?", params + [page_size, offset]).fetchall()
        return [dict(r) for r in rows], total

def delete_n8n_log(log_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM n8n_logs WHERE id = ?", (log_id,))
        conn.commit()


def list_studies(q: str = "", page: int = 1, page_size: int = 10):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='studies_history'")
    if not cur.fetchone():
        conn.close()
        return {"items": [], "total": 0, "page": page, "page_size": page_size}
    offset = max(0, (page - 1) * page_size)
    if q.strip():
        like = f"%{q.strip()}%"
        cur.execute("SELECT COUNT(*) FROM studies_history WHERE patient_name LIKE ? OR patient_mrn LIKE ? OR primary_finding LIKE ?", (like, like, like))
        total = cur.fetchone()[0]
        cur.execute("SELECT * FROM studies_history WHERE patient_name LIKE ? OR patient_mrn LIKE ? OR primary_finding LIKE ? ORDER BY id DESC LIMIT ? OFFSET ?", (like, like, like, page_size, offset))
    else:
        cur.execute("SELECT COUNT(*) FROM studies_history")
        total = cur.fetchone()[0]
        cur.execute("SELECT * FROM studies_history ORDER BY id DESC LIMIT ? OFFSET ?", (page_size, offset))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return {"items": rows, "total": total, "page": page, "page_size": page_size}
