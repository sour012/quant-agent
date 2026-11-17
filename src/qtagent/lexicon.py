"""财经新闻情感词典（轻量、可解释、零外部依赖）。

相比直接调用大模型：词典法在回测中可完全复现、可审计、无调用成本，
对「利好/利空」这类高频财经措辞的判别也足够稳定。
"""

from __future__ import annotations

import math
import re

POSITIVE: dict[str, float] = {
    "上涨": 1.0, "大涨": 1.6, "涨停": 1.8, "新高": 1.5, "反弹": 1.0, "回升": 0.9,
    "利好": 1.5, "超预期": 1.4, "增长": 1.0, "回暖": 1.1, "复苏": 1.2, "向好": 1.1,
    "突破": 1.0, "放量": 0.6, "盈利": 0.9, "增持": 1.1, "回购": 1.0, "提振": 1.2,
    "降准": 1.3, "降息": 1.3, "宽松": 1.1, "刺激": 1.0, "加仓": 0.9, "买入": 0.8,
    "强劲": 1.1, "改善": 0.9, "创新高": 1.6, "扩大": 0.6, "扩内需": 1.0, "稳增长": 1.1,
    "签约": 0.7, "中标": 0.9, "涨价": 0.8, "上调": 1.0, "达成": 0.7, "合作": 0.5,
}

NEGATIVE: dict[str, float] = {
    "下跌": 1.0, "大跌": 1.6, "跌停": 1.8, "新低": 1.5, "回落": 0.9, "回调": 0.8,
    "利空": 1.5, "低于预期": 1.4, "下滑": 1.0, "萎缩": 1.2, "衰退": 1.5, "承压": 0.9,
    "风险": 0.8, "违约": 1.5, "爆雷": 1.7, "抛售": 1.4, "减持": 1.0, "加息": 1.2,
    "收紧": 1.0, "亏损": 1.2, "缩量": 0.5, "恐慌": 1.4, "危机": 1.5, "制裁": 1.4,
    "关税": 1.0, "摩擦": 1.0, "下调": 1.0, "裁员": 1.1, "破产": 1.6, "跳水": 1.5,
    "杀跌": 1.3, "警示": 0.8, "处罚": 1.0, "退市": 1.4, "调查": 0.7, "冲突": 0.9,
}

_POS_PATTERN = re.compile("|".join(sorted(map(re.escape, POSITIVE), key=len, reverse=True)))
_NEG_PATTERN = re.compile("|".join(sorted(map(re.escape, NEGATIVE), key=len, reverse=True)))


def score_text(text: str) -> float:
    """返回 [-1, 1] 的情感分：正数偏利好、负数偏利空。"""
    if not text:
        return 0.0
    pos = sum(POSITIVE[m.group(0)] for m in _POS_PATTERN.finditer(text))
    neg = sum(NEGATIVE[m.group(0)] for m in _NEG_PATTERN.finditer(text))
    if pos == 0 and neg == 0:
        return 0.0
    return math.tanh((pos - neg) / 3.0)


def label_of(score: float, threshold: float = 0.15) -> str:
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"
