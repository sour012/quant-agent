"""数据库访问层：建表、幂等写入、多维检索。

幂等写入的实现方式是「按主键先删后插」，跨 SQLite / MySQL 都成立，
也不会因为重复跑 ETL 而出现脏数据。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Sequence

import pandas as pd
from sqlalchemy import DateTime, and_, create_engine, select, text
from sqlalchemy.engine import Engine

from ..config import Config
from .schema import ALL_TABLES, AUX_TABLE_DDL, fact_market_daily, fact_news, fact_trade_monthly, metadata

logger = logging.getLogger(__name__)


def get_engine(cfg: Config) -> Engine:
    return create_engine(cfg.db_url, future=True)


def init_schema(engine: Engine, with_view: bool = True) -> None:
    metadata.create_all(engine)
    if with_view and engine.dialect.name == "sqlite":
        with engine.begin() as conn:
            for ddl in AUX_TABLE_DDL:
                conn.execute(text(ddl))
    logger.info("表结构就绪: %s", [t.name for t in ALL_TABLES])


def _prepare_records(df: pd.DataFrame, table, cols: Sequence[str]) -> list[dict[str, Any]]:
    work = df.loc[:, list(cols)].copy()
    for col in table.columns:
        if col.name in work.columns and isinstance(col.type, DateTime):
            work[col.name] = pd.to_datetime(work[col.name], errors="coerce")
    work = work.astype(object).where(pd.notnull(work), None)
    records: list[dict[str, Any]] = []
    for row in work.to_dict("records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                value = value.to_pydatetime()
            clean[key] = value
        records.append(clean)
    return records


def upsert_df(engine: Engine, table, df: pd.DataFrame, key_cols: Sequence[str]) -> int:
    """按主键覆盖写入，返回写入行数。"""
    if df is None or df.empty:
        return 0
    cols = [c.name for c in table.columns if c.name in df.columns]
    missing = [k for k in key_cols if k not in cols]
    if missing:
        raise KeyError(f"写入 {table.name} 缺少主键列: {missing}")

    records = _prepare_records(df, table, cols)
    with engine.begin() as conn:
        keys = df.loc[:, list(key_cols)].drop_duplicates().astype(object).where(lambda x: x.notnull(), None)
        for key_values in keys.itertuples(index=False, name=None):
            cond = and_(*[table.c[c] == v for c, v in zip(key_cols, key_values)])
            conn.execute(table.delete().where(cond))
        if records:
            conn.execute(table.insert(), records)
    return len(records)


# ---------------- 读取 ----------------
def load_market(
    engine: Engine, index_code: str, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    stmt = select(fact_market_daily).where(fact_market_daily.c.index_code == index_code)
    if start:
        stmt = stmt.where(fact_market_daily.c.trade_date >= start)
    if end:
        stmt = stmt.where(fact_market_daily.c.trade_date <= end)
    stmt = stmt.order_by(fact_market_daily.c.trade_date)
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn)
    return df.reset_index(drop=True)


def load_trade(engine: Engine, region_code: str = "TOTAL") -> pd.DataFrame:
    stmt = (
        select(fact_trade_monthly)
        .where(fact_trade_monthly.c.region_code == region_code)
        .order_by(fact_trade_monthly.c.trade_month)
    )
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn)
    return df.reset_index(drop=True)


def load_news(engine: Engine, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    stmt = select(fact_news)
    if start:
        stmt = stmt.where(fact_news.c.publish_time >= start)
    if end:
        stmt = stmt.where(fact_news.c.publish_time < end)
    stmt = stmt.order_by(fact_news.c.publish_time)
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn)
    if not df.empty:
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    return df.reset_index(drop=True)


# ---------------- 多维检索（简历里的「时段/口径/流向」） ----------------
_TRADE_FIELD_MAP: dict[tuple[str, str], dict[str, str]] = {
    ("export", "monthly"): {"value": "export_usd", "yoy": "export_yoy", "mom": "export_mom"},
    ("import", "monthly"): {"value": "import_usd", "yoy": "import_yoy", "mom": "import_mom"},
    ("balance", "monthly"): {"value": "balance_usd", "yoy": "export_yoy", "mom": "export_mom"},
    ("export", "accum"): {"value": "export_accum_usd", "yoy": "export_yoy", "mom": "export_mom"},
    ("import", "accum"): {"value": "import_accum_usd", "yoy": "import_yoy", "mom": "import_mom"},
}


def query_trade(
    engine: Engine,
    start_month: str | None = None,
    end_month: str | None = None,
    flow: str = "export",
    calibre: str = "monthly",
    region_code: str = "TOTAL",
) -> pd.DataFrame:
    """按时间区间 + 流向 + 口径检索贸易数据。

    flow: export | import | balance
    calibre: monthly | accum
    """
    key = (flow.lower(), calibre.lower())
    if key not in _TRADE_FIELD_MAP:
        raise ValueError(f"不支持的检索维度: flow={flow}, calibre={calibre}")
    fields = _TRADE_FIELD_MAP[key]

    sql = f"""
        SELECT trade_month,
               {fields['value']} AS value_usd,
               {fields['yoy']}     AS yoy,
               {fields['mom']}     AS mom
        FROM fact_trade_monthly
        WHERE region_code = :region
          AND (:start_month IS NULL OR trade_month >= :start_month)
          AND (:end_month   IS NULL OR trade_month <= :end_month)
        ORDER BY trade_month
    """
    with engine.connect() as conn:
        df = pd.read_sql(
            text(sql),
            conn,
            params={"region": region_code, "start_month": start_month, "end_month": end_month},
        )
    return df


def save_signals(engine: Engine, records: Iterable[dict[str, Any]], run_id: str | None = None) -> int:
    from .schema import fact_signal_daily

    rows = []
    for rec in records:
        row = dict(rec)
        for key in ('contributions', 'reasons'):
            if isinstance(row.get(key), (dict, list)):
                row[key] = json.dumps(row[key], ensure_ascii=False)
        if run_id:
            row.setdefault("run_id", run_id)
        rows.append(row)
    if not rows:
        return 0
    return upsert_df(engine, fact_signal_daily, pd.DataFrame(rows), ["trade_date"])


def count_rows(engine: Engine, table_name: str) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())
