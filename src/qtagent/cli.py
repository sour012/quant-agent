"""命令行入口：`python -m qtagent <command>`"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import setup_logging
from .backtest import BacktestEngine, build_markdown, compute_metrics, plot_equity
from .config import Config
from .etl import build_db
from .orchestrator import TradingPipeline
from .storage import get_engine, query_trade, save_signals

logger = logging.getLogger("qtagent.cli")

FMT = "{:<14}{:>14}"


def _print_metrics(metrics: dict) -> None:
    pct = {"total_return", "cagr", "ann_vol", "max_drawdown", "exposure", "win_rate",
           "benchmark_return", "benchmark_max_drawdown", "excess_return", "excess_cagr"}
    for key in [
        "total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "calmar",
        "benchmark_return", "excess_return", "exposure", "turnover", "trades", "win_rate",
    ]:
        if key not in metrics:
            continue
        value = metrics[key]
        if key in pct:
            print(FMT.format(key, f"{value:.2%}"))
        elif key in {"trades"}:
            print(FMT.format(key, f"{int(value)}"))
        else:
            print(FMT.format(key, f"{value:.3f}"))


# ---------------- 子命令 ----------------
def cmd_build_db(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    summary = build_db(cfg, export_snapshot=not args.no_snapshot)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    engine = get_engine(cfg)
    df = query_trade(
        engine,
        start_month=args.start,
        end_month=args.end,
        flow=args.flow,
        calibre=args.calibre,
    )
    if df.empty:
        print("没有匹配的数据")
        return 0
    print(df.tail(args.limit).to_string(index=False))
    return 0


def cmd_signal(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    pipeline = TradingPipeline(cfg)
    date = args.date or pipeline.dates[-1].strftime("%Y-%m-%d")
    payload = pipeline.explain(date)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    d = payload["decision"]
    print(f"\n{'='*74}")
    print(f"交易日 {payload['trade_date']}   信号：{d['action']}   综合分 {d['score']:+.3f}   "
          f"目标仓位 {d['target_position']:.0%}（上限 {d['position_cap']:.0%}）")
    print("=" * 74)
    for name, info in payload["agents"].items():
        print(f"\n[{name}] score={info['score']:+.3f} conf={info['confidence']:.2f}")
        for r in info["reasons"]:
            print(f"   - {r}")
    print("\n归因：")
    for r in d["reasons"]:
        print(f"   * {r}")
    print()
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    pipeline = TradingPipeline(cfg)
    engine = BacktestEngine(pipeline, cfg, start=args.start, end=args.end)
    result = engine.run()

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or "backtest"
    daily_path = cfg.report_dir / f"{tag}_daily.csv"
    png_path = cfg.report_dir / f"{tag}_equity.png"
    md_path = cfg.report_dir / f"{tag}.md"

    result.daily.to_csv(daily_path, index=False, encoding="utf-8")
    if not result.trades.empty:
        result.trades.to_csv(cfg.report_dir / f"{tag}_trades.csv", index=False, encoding="utf-8")
    plot_equity(result.daily, png_path, dpi=int(cfg.output.get("figure_dpi", 150)))

    extra = {
        "数据源": cfg.data.get("provider"),
        "标的": f"{cfg.data.get('index_code')} 沪深300",
        "Agent 权重": " / ".join(f"{k}={v:.2f}" for k, v in cfg.weights.items()),
        "成本假设": f"佣金 {cfg.backtest.get('commission'):.4%} + 滑点 {cfg.backtest.get('slippage'):.4%}（单边）",
        "执行方式": "T 日收盘出信号，T+1 开盘成交",
    }
    md_path.write_text(build_markdown(result, f"回测报告 · {tag}", extra), encoding="utf-8")

    if args.save_signals:
        save_signals(get_engine(cfg), result.decisions, run_id=tag)

    print(f"\n回测区间 {result.metrics.get('start')} ~ {result.metrics.get('end')}")
    _print_metrics(result.metrics)
    print(f"\n明细：{daily_path}\n净值图：{png_path}\n报告：{md_path}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    """权重网格搜索：样本内选参、样本外验证，避免只看全样本的过拟合结论。"""
    cfg = Config.load(args.config)
    pipeline = TradingPipeline(cfg)
    dates = pipeline.dates
    split_idx = int(len(dates) * args.in_sample)
    split_date = dates[split_idx].strftime("%Y-%m-%d")
    first = dates[pipeline.WARMUP_BARS].strftime("%Y-%m-%d")
    last = dates[-1].strftime("%Y-%m-%d")

    grid = [
        {"technical": t, "sentiment": s, "macro": m, "risk": r}
        for t in (0.30, 0.40, 0.50)
        for s in (0.10, 0.15)
        for m in (0.15, 0.25)
        for r in (0.15, 0.25)
    ]
    base_weights = cfg.weights
    rows = []
    for i, w in enumerate(grid, 1):
        total = sum(w.values())
        cfg.raw["agents"]["decision"]["weights"] = {k: v / total for k, v in w.items()}
        ins = BacktestEngine(pipeline, cfg, start=first, end=split_date).run().metrics
        oos = BacktestEngine(pipeline, cfg, start=split_date, end=last).run().metrics
        rows.append(
            {
                "technical": w["technical"], "sentiment": w["sentiment"],
                "macro": w["macro"], "risk": w["risk"],
                "in_cagr": ins.get("cagr", 0.0), "in_sharpe": ins.get("sharpe", 0.0),
                "in_mdd": ins.get("max_drawdown", 0.0),
                "oos_cagr": oos.get("cagr", 0.0), "oos_sharpe": oos.get("sharpe", 0.0),
                "oos_mdd": oos.get("max_drawdown", 0.0),
            }
        )
        logger.info("组合 %d/%d 完成", i, len(grid))

    cfg.raw["agents"]["decision"]["weights"] = base_weights
    df = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)
    out = cfg.report_dir / "tuning.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    print(f"\n样本内 {first} ~ {split_date}，样本外 {split_date} ~ {last}，共 {len(df)} 组权重\n")
    print(df.head(args.top).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n结果已保存：{out}")
    print("提示：样本外指标的排序才是选参依据；若两段结论相反，说明权重不稳，应减少对单一 Agent 的依赖。")
    return 0


# ---------------- 解析 ----------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qtagent", description="五 Agent 协作的量化交易决策系统"
    )
    parser.add_argument("--config", default=None, help="配置文件路径（默认 config/config.yaml）")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-db", help="抓取数据、清洗并写入数据库")
    p.add_argument("--no-snapshot", action="store_true", help="不导出离线快照 CSV")
    p.set_defaults(func=cmd_build_db)

    p = sub.add_parser("signal", help="查看某个交易日的五 Agent 决策与归因")
    p.add_argument("--date", default=None, help="YYYY-MM-DD，缺省为最新交易日")
    p.add_argument("--json", action="store_true", help="输出 JSON 结构")
    p.set_defaults(func=cmd_signal)

    p = sub.add_parser("backtest", help="运行策略回测并生成报告")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--tag", default=None, help="报告文件名前缀")
    p.add_argument("--save-signals", action="store_true", help="把每日信号写入 fact_signal_daily")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("tune", help="Agent 权重网格搜索（样本内选参 + 样本外验证）")
    p.add_argument("--in-sample", type=float, default=0.7, help="样本内占比，默认 0.7")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("query", help="按时间/流向/口径多维检索贸易数据")
    p.add_argument("--flow", default="export", choices=["export", "import", "balance"])
    p.add_argument("--calibre", default="monthly", choices=["monthly", "accum"])
    p.add_argument("--start", default=None, help="起始月份 YYYY-MM")
    p.add_argument("--end", default=None, help="结束月份 YYYY-MM")
    p.add_argument("--limit", type=int, default=12)
    p.set_defaults(func=cmd_query)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("已中断")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("执行失败: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
