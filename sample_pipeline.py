"""数据飞轮五算子脚本（必带 Model API 语义评测）。

输入：JSON 数组或 JSONL。每条记录应至少包含：
sessionId、messageId、conversations、success、outputTokens。

运行前配置 .env（或环境变量）：
    MODEL_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
    MODEL_API_KEY=sk-xxx
    MODEL_NAME=qwen3.5-plus
    MODEL_TIMEOUT=120

    python sample_pipeline.py --input samples.jsonl --output result.jsonl
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


ASSISTANT_ROLES = {"assistant", "bot", "ai", "model"}
USER_ROLES = {"user", "human"}
NEGATIVE_THUMBS = {"down", "negative", "bad", "dislike", "0", "false"}
POSITIVE_THUMBS = {"up", "positive", "good", "like", "1", "true"}


def load_dotenv(path: str | None = None) -> None:
    dotenv_path = Path(path or ".env")
    if not dotenv_path.is_file():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON 顶层必须是数组")
        records = data
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    invalid = [type(r).__name__ for r in records if not isinstance(r, dict)]
    if invalid:
        raise ValueError(f"非对象记录：{invalid[:3]}")
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def redact_text(value: str) -> str:
    """脱敏：手机号、身份证、银行卡、邮箱、token。"""
    value = re.sub(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)", "[PHONE]", value)
    value = re.sub(r"(?<!\d)\d{15}(?!\d)", "[ID_CARD]", value)
    value = re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "[ID_CARD]", value)
    value = re.sub(r"(?<!\d)\d{16,19}(?!\d)", "[BANK_CARD]", value)
    value = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", value)
    value = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[TOKEN]", value)
    value = re.sub(r"(?i)(api[-_ ]?key\s*[:=]\s*)[^\s,;]+", r"\1[TOKEN]", value)
    return value


def normalize_conversations(value: Any, redact: bool) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("from", item.get("role", ""))).strip().lower()
        content = item.get("value", item.get("content", "")) or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if redact:
            content = redact_text(content)
        if role in ASSISTANT_ROLES:
            role = "assistant"
        elif role in USER_ROLES:
            role = "user"
        if role and content.strip():
            result.append({"from": role, "value": content.strip()})
    return result


def first_value(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        v = record.get(name)
        if v not in (None, ""):
            return v
    return None


def numeric(record: dict[str, Any], name: str) -> float | None:
    v = record.get(name)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def assistant_text(record: dict[str, Any]) -> str:
    return "\n".join(
        m["value"] for m in record.get("conversations", []) if m.get("from") == "assistant"
    ).strip()


# ====================== ① 规整 ======================

def prepare_dataset(raw: list[dict[str, Any]], redact: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """算子 1：字段规整、去重、脱敏。只丢结构无效；业务异常留 warnings 给后续算子。"""
    prepared: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for idx, orig in enumerate(raw):
        r = copy.deepcopy(orig)
        sid = first_value(r, "sessionId", "sessionID")
        mid = first_value(r, "messageId", "messageID")
        convs = normalize_conversations(r.get("conversations"), redact)

        # 结构无效 → 直接丢
        reasons: list[str] = []
        if not sid:
            reasons.append("missing_sessionId")
        if not mid:
            reasons.append("missing_messageId")
        if not convs:
            reasons.append("empty_conversations")
        key = f"{sid}:{mid}" if sid and mid else stable_hash(convs)
        if key in seen:
            reasons.append("duplicate")
        if reasons:
            rejected.append({"inputIndex": idx, "reasons": reasons, "messageId": mid})
            continue

        seen.add(key)
        r["sessionId"] = str(sid)
        r["messageId"] = str(mid)
        r["conversations"] = convs
        r["sampleId"] = stable_hash({"sessionId": sid, "messageId": mid})[:20]

        # 业务异常打 warnings，不丢
        warnings: list[str] = []
        if r.get("success") is False:
            warnings.append("generation_failed")
        if not any(m["from"] == "assistant" for m in convs):
            warnings.append("missing_assistant_answer")
        if warnings:
            r["_prepareWarnings"] = warnings

        r.pop("traceId", None)
        r.pop("userId", None)
        if isinstance(r.get("userFeedback"), dict) and redact:
            fb = r["userFeedback"]
            if isinstance(fb.get("comment"), str):
                fb["comment"] = redact_text(fb["comment"])
        prepared.append(r)

    return prepared, rejected


# ====================== Model API ======================

class ModelApiClient:
    """OpenAI-compatible 语义评测客户端。"""

    def __init__(self, url: str, api_key: str | None, model: str, timeout: int = 120):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def evaluate(self, record: dict[str, Any]) -> dict[str, Any]:
        prompt = {
            "task": "评价样本是否值得进入高质量数据候选集。只返回 JSON，不要 Markdown。",
            "output_schema": {
                "task_satisfaction": "0-1，回答是否满足用户任务",
                "answer_quality": "0-1，正确性、完整性、可用性综合",
                "complexity": "0-1，任务约束和推理复杂度",
                "reason": "不超过100字的简短理由",
            },
            "conversations": record.get("conversations", []),
        }
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "你是数据质量评估器。必须输出合法 JSON，不要输出 Markdown。"},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            result = _parse_json(content)
            return {
                "status": "ok",
                "taskSatisfaction": _clamp(result.get("task_satisfaction")),
                "answerQuality": _clamp(result.get("answer_quality")),
                "complexity": _clamp(result.get("complexity")),
                "reason": str(result.get("reason", ""))[:200],
            }
        except Exception as exc:
            return {"status": "error", "reason": f"model_api_error: {exc}"}


def _parse_json(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(str(i.get("text", "")) if isinstance(i, dict) else str(i) for i in content)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(content).strip(), flags=re.IGNORECASE | re.DOTALL)
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        raise ValueError("not a JSON object")
    return json.loads(text[s : e + 1])


def _clamp(v: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return None


# ====================== 阈值 ======================

THRESHOLD_SLOW_REQUEST_MS = 15_000
THRESHOLD_SLOW_TTFT_MS = 5_000
SEMANTIC_QUALIFIED_MIN = 0.6
SEMANTIC_UNQUALIFIED_MAX = 0.4


# ====================== ③ 反馈分析 ======================

def _extract_feedback(record: dict[str, Any]) -> tuple[str, str | None]:
    fb = record.get("userFeedback")
    if not isinstance(fb, dict):
        return "unknown", None
    thumbs = str(fb.get("thumbs", "")).strip().lower()
    if thumbs in POSITIVE_THUMBS:
        return "positive", thumbs
    if thumbs in NEGATIVE_THUMBS:
        return "negative", thumbs
    return "unknown", thumbs or None


def _events(record: dict[str, Any]) -> set[str]:
    ev = record.get("userInteractionEvents")
    if not isinstance(ev, list):
        return set()
    return {str(e).strip().lower() for e in ev if e is not None}


def analyze_feedback(record: dict[str, Any], client: ModelApiClient) -> dict[str, Any]:
    """算子 3：性能指标 + 用户反馈 + Model API 语义评分。"""
    fb_status, thumbs = _extract_feedback(record)
    events = _events(record)
    analysis: dict[str, Any] = {
        "feedbackStatus": fb_status,
        "feedbackThumbs": thumbs,
        "interactionSignals": sorted(events),
        "performanceFlags": [],
    }
    dur = numeric(record, "durationMs")
    ttft = numeric(record, "ttftMs")
    out_tok = numeric(record, "outputTokens")
    if dur is not None and dur > THRESHOLD_SLOW_REQUEST_MS:
        analysis["performanceFlags"].append("slow_request")
    if ttft is not None and ttft > THRESHOLD_SLOW_TTFT_MS:
        analysis["performanceFlags"].append("slow_first_token")
    if out_tok is not None and out_tok <= 0:
        analysis["performanceFlags"].append("empty_output")

    # 语义评测：只对 success=true 且有回答的样本调用
    if record.get("success") is True and assistant_text(record):
        analysis["semanticEvaluation"] = client.evaluate(record)
    else:
        analysis["semanticEvaluation"] = {"status": "skipped", "reason": "not_success_or_empty_answer"}
    return analysis


# ====================== ② 难例挖掘 ======================

def find_hard_case(record: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    """算子 2：标记难例。"""
    reasons: list[str] = []
    if record.get("success") is False:
        reasons.append("generation_failed")
    if analysis["feedbackStatus"] == "negative":
        reasons.append("negative_feedback")
    if "stop_generation" in analysis["interactionSignals"] or "stop_generate" in analysis["interactionSignals"]:
        reasons.append("stopped_by_user")
    if "regenerate" in analysis["interactionSignals"]:
        reasons.append("regenerated")
    if "empty_output" in analysis["performanceFlags"]:
        reasons.append("empty_output")
    sem = analysis["semanticEvaluation"]
    if sem.get("status") == "ok":
        if sem.get("taskSatisfaction") is not None and sem["taskSatisfaction"] < SEMANTIC_QUALIFIED_MIN:
            reasons.append("low_task_satisfaction")
        if sem.get("answerQuality") is not None and sem["answerQuality"] < SEMANTIC_QUALIFIED_MIN:
            reasons.append("low_answer_quality")
    return {"isHardExample": bool(reasons), "reasons": reasons}


# ====================== ④ 候选筛选 ======================

def classify_candidate(record: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    """算子 4：qualified / unqualified / pending 三路分流。

    硬拒绝（直接 unqualified）：
        request_not_successful / empty_assistant_answer / empty_output
    pending：
        - 语义在 0.4-0.6 中间区间
        - 正反馈 + 语义 < 0.4（矛盾信号）
        - 负反馈
    语义 >= 0.6 且 quality >= 0.6 且无 pending 原因 → qualified
    语义 < 0.4 → unqualified
    """
    reject: list[str] = []
    pending: list[str] = []

    # 硬拒绝
    if record.get("success") is not True:
        reject.append("request_not_successful")
    if not assistant_text(record):
        reject.append("empty_assistant_answer")
    if numeric(record, "outputTokens") is not None and numeric(record, "outputTokens") <= 0:
        reject.append("empty_output")
    if reject:
        return {"filterResult": "unqualified", "filterReasons": reject}

    # 负反馈 → pending
    if analysis["feedbackStatus"] == "negative":
        pending.append("negative_feedback")

    sem = analysis["semanticEvaluation"]
    if sem.get("status") == "ok":
        t, q = sem.get("taskSatisfaction"), sem.get("answerQuality")

        # 矛盾信号：正反馈但语义差
        if analysis["feedbackStatus"] == "positive" and (t is not None and t < SEMANTIC_UNQUALIFIED_MAX or q is not None and q < SEMANTIC_UNQUALIFIED_MAX):
            pending.append("positive_but_low_semantic")

        qualified = t is not None and q is not None and t >= SEMANTIC_QUALIFIED_MIN and q >= SEMANTIC_QUALIFIED_MIN
        unqualified = t is not None and q is not None and (t < SEMANTIC_UNQUALIFIED_MAX or q < SEMANTIC_UNQUALIFIED_MAX)

        if qualified and not pending:
            return {"filterResult": "qualified", "filterReasons": ["technical_checks_passed", "semantic_scores_passed"]}
        if unqualified and not pending:
            return {"filterResult": "unqualified", "filterReasons": ["semantic_scores_below_threshold"]}
        return {"filterResult": "pending", "filterReasons": ["semantic_middle_or_conflict"] + pending}

    # 语义评测被跳过（success=false / 空回答）
    if sem.get("status") == "skipped":
        # 硬拒绝已在上方处理过，到这里说明 success=true 但 sem 异常，不太可能
        return {"filterResult": "pending", "filterReasons": ["semantic_skipped"] + pending}

    # Model API 调用 error
    return {"filterResult": "pending", "filterReasons": ["semantic_api_error"] + pending}


# ====================== ⑤ 增量高价值 ======================

def scene_key(record: dict[str, Any]) -> str:
    return "|".join(str(record.get(f) or "unknown") for f in ("systemName", "category", "X-Ai-Scene-Name"))


def select_high_value(record: dict[str, Any], analysis: dict[str, Any],
                      candidate: dict[str, Any], scene_counts: Counter[str], rare_threshold: int) -> dict[str, Any]:
    """算子 5：从 qualified 中选增量高价值样本。"""
    if candidate.get("filterResult") != "qualified":
        return {"isHighValue": False, "reasons": ["not_qualified"]}

    reasons: list[str] = []
    events = analysis["interactionSignals"]
    if analysis["feedbackStatus"] == "positive":
        reasons.append("positive_feedback")
    if "copy" in events or "share" in events or "adopt" in events:
        reasons.append("usage_intent")
    if "regenerate" in events:
        reasons.append("successful_after_regeneration")
    sem = analysis["semanticEvaluation"]
    if sem.get("complexity") is not None and sem["complexity"] >= 0.7:
        reasons.append("complex_task")
    if scene_counts[scene_key(record)] <= rare_threshold:
        reasons.append("rare_production_scene")
    return {"isHighValue": bool(reasons), "reasons": reasons}


# ====================== 主流程 ======================

def run_pipeline(raw: list[dict[str, Any]], redact: bool, client: ModelApiClient, rare_threshold: int
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared, rejected = prepare_dataset(raw, redact)
    staged: list[dict[str, Any]] = []
    for r in prepared:
        analysis = analyze_feedback(r, client)
        hard_case = find_hard_case(r, analysis)
        candidate = classify_candidate(r, analysis)
        staged.append({"record": r, "analysis": analysis, "hardCase": hard_case, "candidate": candidate})

    scene_counts = Counter(
        scene_key(s["record"]) for s in staged if s["candidate"]["filterResult"] == "qualified"
    )
    output: list[dict[str, Any]] = []
    for s in staged:
        high_value = select_high_value(s["record"], s["analysis"], s["candidate"], scene_counts, rare_threshold)
        result = copy.deepcopy(s["record"])
        result.pop("_prepareWarnings", None)
        result["pipeline"] = {
            "feedbackAnalysis": s["analysis"],
            "hardCase": s["hardCase"],
            "candidate": s["candidate"],
            "highValue": high_value,
        }
        output.append(result)
    return output, rejected


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="数据飞轮五算子（必带 Model API）")
    parser.add_argument("--input", required=True, help="输入 JSON 或 JSONL")
    parser.add_argument("--output", required=True, help="输出 JSONL")
    parser.add_argument("--rejected-output", help="规整阶段被过滤的记录输出")
    parser.add_argument("--model-api-url", default=os.getenv("MODEL_API_URL"))
    parser.add_argument("--model-api-key", default=os.getenv("MODEL_API_KEY"))
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "qwen3.5-plus"))
    parser.add_argument("--model-timeout", type=int, default=int(os.getenv("MODEL_TIMEOUT", "120")))
    parser.add_argument("--rare-threshold", type=int, default=3, help="同场景 qualified <= N 算稀缺")
    parser.add_argument("--no-redact", action="store_true", help="关闭脱敏（仅调试）")
    args = parser.parse_args()

    if not args.model_api_url:
        print("缺少 MODEL_API_URL。请在 .env 或环境变量中配置。", file=sys.stderr)
        return 2

    client = ModelApiClient(args.model_api_url, args.model_api_key, args.model_name, args.model_timeout)
    raw = load_records(Path(args.input))
    output, rejected = run_pipeline(raw, redact=not args.no_redact, client=client, rare_threshold=max(args.rare_threshold, 0))

    write_jsonl(Path(args.output), output)
    if args.rejected_output:
        write_jsonl(Path(args.rejected_output), rejected)

    counts = Counter(item["pipeline"]["candidate"]["filterResult"] for item in output)
    hv = sum(1 for item in output if item["pipeline"]["highValue"]["isHighValue"])
    print(json.dumps({
        "input": len(raw), "prepared": len(output), "rejectedInPreparation": len(rejected),
        "candidateCounts": dict(counts), "highValue": hv, "output": args.output,
    }, ensure_ascii=False))
    if rejected:
        print("规整阶段被过滤：", json.dumps(rejected[:10], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
