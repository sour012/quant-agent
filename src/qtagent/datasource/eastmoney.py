"""东方财富公开数据接口客户端。

职责：只负责「把外部数据取回来并整理成规范 DataFrame」，不做清洗与落库。
- 统一重试与超时，避免偶发网络抖动打断 ETL
- 所有返回列名、单位、日期格式统一，方便下游入库
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class EastMoneyClient:
    HIS_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    DC_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    NEWS_URL = "https://np-listapi.eastmoney.com/comm/web/getFastNewsList"

    def __init__(self, timeout: int = 20, retries: int = 3, retry_sleep: float = 1.5) -> None:
        self.timeout = timeout
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA, "Referer": "https://finance.eastmoney.com/"})

    # ---------------- 基础请求 ----------------
    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    raise ValueError("返回内容不是 JSON 对象")
                return data
            except Exception as exc:  # noqa: BLE001 - 网络层需要兜住所有异常
                last_err = exc
                logger.warning("请求失败(%d/%d) %s: %s", attempt, self.retries, url, exc)
                if attempt < self.retries:
                    time.sleep(self.retry_sleep * attempt)
        raise RuntimeError(f"请求最终失败: {url} ({last_err})") from last_err

    # ---------------- 海关进出口（月度） ----------------
    def customs_monthly(self, page_size: int = 100) -> pd.DataFrame:
        """中国海关月度进出口。

        EXIT_BASE / IMPORT_BASE 单位为千美元，统一折算为美元。
        """
        rows: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = self._get(
                self.DC_URL,
                {
                    "reportName": "RPT_ECONOMY_CUSTOMS",
                    "columns": "ALL",
                    "pageNumber": page,
                    "pageSize": page_size,
                    "sortColumns": "REPORT_DATE",
                    "sortTypes": -1,
                    "source": "WEB",
                    "client": "WEB",
                },
            )
            result = payload.get("result") or {}
            data = result.get("data") or []
            if not data:
                break
            rows.extend(data)
            if page >= int(result.get("pages") or 1):
                break
            page += 1

        if not rows:
            raise RuntimeError("海关进出口数据为空")

        raw = pd.DataFrame(rows)
        out = pd.DataFrame(
            {
                "trade_month": pd.to_datetime(raw["REPORT_DATE"]).dt.strftime("%Y-%m"),
                "export_usd": pd.to_numeric(raw["EXIT_BASE"], errors="coerce") * 1_000,
                "import_usd": pd.to_numeric(raw["IMPORT_BASE"], errors="coerce") * 1_000,
                "export_yoy": pd.to_numeric(raw["EXIT_BASE_SAME"], errors="coerce"),
                "import_yoy": pd.to_numeric(raw["IMPORT_BASE_SAME"], errors="coerce"),
                "export_mom": pd.to_numeric(raw["EXIT_BASE_SEQUENTIAL"], errors="coerce"),
                "import_mom": pd.to_numeric(raw["IMPORT_BASE_SEQUENTIAL"], errors="coerce"),
                "export_accum_usd": pd.to_numeric(raw["EXIT_ACCUMULATE"], errors="coerce") * 1_000,
                "import_accum_usd": pd.to_numeric(raw["IMPORT_ACCUMULATE"], errors="coerce") * 1_000,
            }
        )
        out = out.dropna(subset=["trade_month"]).drop_duplicates(subset=["trade_month"])
        out = out.sort_values("trade_month").reset_index(drop=True)
        out["balance_usd"] = out["export_usd"] - out["import_usd"]
        return out

    # ---------------- 指数日线 ----------------
    def index_daily(
        self, code: str = "000300", start: str = "2019-01-01", end: str = "20500101"
    ) -> pd.DataFrame:
        secid = self._secid(code)
        payload = self._get(
            self.HIS_URL,
            {
                "secid": secid,
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57",
                "klt": 101,   # 日线
                "fqt": 1,     # 前复权
                "beg": start.replace("-", ""),
                "end": end.replace("-", ""),
                "lmt": 100000,
            },
        )
        data = payload.get("data") or {}
        klines = data.get("klines") or []
        if not klines:
            raise RuntimeError(f"指数 {code} 日线为空（secid={secid}）")
        rows = [line.split(",") for line in klines]
        df = pd.DataFrame(
            rows, columns=["trade_date", "open", "close", "high", "low", "volume", "amount"]
        )
        for col in ["open", "close", "high", "low", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.insert(0, "index_code", code)
        return df

    def fx_daily(self, pair: str = "USDCNH", start: str = "2019-01-01") -> pd.DataFrame:
        secid = "133." + pair.upper()
        payload = self._get(
            self.HIS_URL,
            {
                "secid": secid,
                "fields1": "f1,f2",
                "fields2": "f51,f53",
                "klt": 101,
                "fqt": 1,
                "beg": start.replace("-", ""),
                "end": "20500101",
                "lmt": 100000,
            },
        )
        klines = (payload.get("data") or {}).get("klines") or []
        if not klines:
            raise RuntimeError(f"汇率 {pair} 数据为空")
        rows = [line.split(",")[:2] for line in klines]
        df = pd.DataFrame(rows, columns=["trade_date", "close"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df.insert(0, "pair", pair.upper())
        return df

    # ---------------- 7x24 财经快讯 ----------------
    def fast_news(self, pages: int = 20, page_size: int = 100) -> pd.DataFrame:
        """按 realSort 游标翻页抓取快讯（由新到旧）。"""
        rows: list[dict[str, Any]] = []
        cursor = ""
        for _ in range(max(1, pages)):
            payload = self._get(
                self.NEWS_URL,
                {
                    "client": "web",
                    "biz": "web_724",
                    "fastColumn": 102,
                    "pageSize": page_size,
                    "sortEnd": cursor,
                    "req_trace": "1",
                },
            )
            data = payload.get("data") or {}
            items = data.get("fastNewsList") or []
            if not items:
                break
            rows.extend(items)
            cursor = str(items[-1].get("realSort") or "")
            if not cursor:
                break
            time.sleep(0.2)

        if not rows:
            return pd.DataFrame(columns=["news_id", "publish_time", "title", "summary", "source"])

        raw = pd.DataFrame(rows)
        out = pd.DataFrame(
            {
                "news_id": raw["code"].astype(str),
                "publish_time": pd.to_datetime(raw["showTime"], errors="coerce"),
                "title": raw["title"].fillna("").astype(str).str.strip(),
                "summary": raw["summary"].fillna("").astype(str).str.strip(),
                "source": "eastmoney_724",
            }
        )
        out = out.dropna(subset=["publish_time"]).drop_duplicates(subset=["news_id"])
        return out.sort_values("publish_time").reset_index(drop=True)

    # ---------------- 内部 ----------------
    @staticmethod
    def _secid(code: str) -> str:
        code = str(code)
        if "." in code:
            return code
        sh_prefix = ("000", "999", "5", "6", "11", "51")
        return ("1." if code.startswith(sh_prefix) else "0.") + code
