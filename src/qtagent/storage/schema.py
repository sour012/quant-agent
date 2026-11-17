"""数据仓库表结构（SQLAlchemy Core，SQLite / MySQL 通用）。

分层思路：
  dim_*    维度表：区域、Agent 元数据
  fact_*   事实表：海关月度、指数日线、汇率日线、新闻、每日信号
事实表全部带 source + updated_at，便于追溯数据来源与增量更新。
"""

from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    func,
)

metadata = MetaData()

dim_region = Table(
    "dim_region",
    metadata,
    Column("region_code", String(16), primary_key=True),
    Column("region_name", String(64), nullable=False),
    Column("region_type", String(16), nullable=False),  # country | bloc | total
    Column("note", String(255)),
)

dim_agent = Table(
    "dim_agent",
    metadata,
    Column("agent_code", String(32), primary_key=True),
    Column("agent_name", String(64), nullable=False),
    Column("responsibility", String(255), nullable=False),
    Column("weight", Float, nullable=False),
)

fact_trade_monthly = Table(
    "fact_trade_monthly",
    metadata,
    Column("trade_month", String(7), primary_key=True),  # YYYY-MM
    Column("region_code", String(16), primary_key=True, default="TOTAL"),
    Column("export_usd", Float),
    Column("import_usd", Float),
    Column("balance_usd", Float),
    Column("export_yoy", Float),
    Column("import_yoy", Float),
    Column("export_mom", Float),
    Column("import_mom", Float),
    Column("export_accum_usd", Float),
    Column("import_accum_usd", Float),
    Column("source", String(32)),
    Column("updated_at", DateTime, server_default=func.now()),
    Index("idx_trade_month", "trade_month"),
)

fact_market_daily = Table(
    "fact_market_daily",
    metadata,
    Column("index_code", String(16), primary_key=True),
    Column("trade_date", String(10), primary_key=True),  # YYYY-MM-DD
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("volume", Float),
    Column("amount", Float),
    Column("source", String(32)),
    Index("idx_market_date", "trade_date"),
)

fact_fx_daily = Table(
    "fact_fx_daily",
    metadata,
    Column("pair", String(12), primary_key=True),
    Column("trade_date", String(10), primary_key=True),
    Column("close", Float),
    Column("source", String(32)),
)

fact_news = Table(
    "fact_news",
    metadata,
    Column("news_id", String(32), primary_key=True),
    Column("publish_time", DateTime, nullable=False),
    Column("title", Text),
    Column("summary", Text),
    Column("source", String(32)),
    Column("sentiment_score", Float),
    Column("sentiment_label", String(16)),
    Index("idx_news_time", "publish_time"),
)

fact_signal_daily = Table(
    "fact_signal_daily",
    metadata,
    Column("trade_date", String(10), primary_key=True),
    Column("action", String(8), nullable=False),        # BUY / SELL / HOLD
    Column("score", Float, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("target_position", Float, nullable=False),
    Column("exec_position", Float),
    Column("equity", Float),
    Column("contributions", Text),                       # JSON: 各 Agent 加权贡献
    Column("reasons", Text),                             # JSON: 可解释文案
    Column("run_id", String(32)),
)

ALL_TABLES = [
    dim_region,
    dim_agent,
    fact_trade_monthly,
    fact_market_daily,
    fact_fx_daily,
    fact_news,
    fact_signal_daily,
]

AUX_TABLE_DDL = [
    # 便于直接写 SQL 做多维检索的中转视图（SQLite 语法）
    """
    CREATE VIEW IF NOT EXISTS v_trade_multi_dim AS
    SELECT t.trade_month,
           t.region_code,
           r.region_name,
           t.export_usd, t.import_usd, t.balance_usd,
           t.export_yoy, t.import_yoy,
           t.export_accum_usd, t.import_accum_usd
    FROM fact_trade_monthly t
    LEFT JOIN dim_region r ON r.region_code = t.region_code
    """,
]

__all__ = ["metadata", "ALL_TABLES", "AUX_TABLE_DDL", "Integer"]
