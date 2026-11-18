"""A small SQLite world with semantic traps, and a read-only executor the agent calls as a tool."""
from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from typing import Any

SCHEMA = """CREATE TABLE customers (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT NOT NULL
);
CREATE TABLE orders (
  id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL,
  amount_cents INTEGER NOT NULL, status TEXT NOT NULL
);"""


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    reference_sql: str


TASKS = (
    Task("paid_total_by_country",
         "Return each country and the sum of amount_cents for paid orders only. "
         "Include only countries with paid orders. Order by country.",
         "SELECT c.country, SUM(o.amount_cents) FROM customers c JOIN orders o "
         "ON o.customer_id=c.id WHERE o.status='paid' GROUP BY c.country ORDER BY c.country"),
    Task("no_paid_orders",
         "Return id and name for every customer who has no paid orders, including "
         "customers with no orders at all. Order by id.",
         "SELECT c.id,c.name FROM customers c WHERE NOT EXISTS "
         "(SELECT 1 FROM orders o WHERE o.customer_id=c.id AND o.status='paid') ORDER BY c.id"),
    Task("largest_paid_order",
         "Return the id and amount_cents of the largest paid order. If amounts tie, "
         "return the smallest order id.",
         "SELECT id,amount_cents FROM orders WHERE status='paid' ORDER BY amount_cents DESC,id ASC LIMIT 1"),
)


def fixture(seed: int) -> sqlite3.Connection:
    """Same traps every time (a cancelled giant order, a tie), different ids and amounts."""
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    countries = ["CA", "US", "GB"]
    rng.shuffle(countries)
    customers = [(i, f"customer_{i}", countries[i % 3]) for i in range(1, 7)]
    rng.shuffle(customers)
    conn.executemany("INSERT INTO customers VALUES (?,?,?)", customers)
    amount = rng.randint(100, 900)
    offset = seed * 100
    rows = [(offset + 1, 1, amount, "paid"), (offset + 2, 1, amount + 10, "paid"),
            (offset + 3, 2, amount + 20, "paid"), (offset + 4, 3, amount + 30, "cancelled"),
            (offset + 5, 4, amount + 99999, "cancelled"), (offset + 6, 5, amount + 500, "paid"),
            (offset + 7, 5, amount + 500, "paid")]
    rng.shuffle(rows)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", rows)
    conn.commit()
    return conn


class QueryRejected(Exception):
    pass


def execute_readonly(conn: sqlite3.Connection, sql: str, *, max_rows: int = 1000) -> list[list[Any]]:
    """SELECT only, bounded work, bounded rows."""
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
    ticks = 0

    def authorize(action, arg1, arg2, *_):
        if action not in allowed:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in {"load_extension", "readfile", "writefile"}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def progress():
        nonlocal ticks
        ticks += 1
        return int(ticks > 100)

    if not sql.strip():
        raise QueryRejected("empty SQL")
    conn.set_authorizer(authorize)
    conn.set_progress_handler(progress, 1000)
    try:
        rows = conn.execute(sql).fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise QueryRejected("row limit exceeded")
        return [list(row) for row in rows]
    except sqlite3.DatabaseError as exc:
        raise QueryRejected(str(exc)) from exc
    finally:
        conn.set_authorizer(None)
        conn.set_progress_handler(None, 0)


def schema_text(conn: sqlite3.Connection) -> str:
    rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return "\n".join(r[0] for r in rows)
