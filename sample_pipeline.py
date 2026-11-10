"""数据飞轮五算子脚本。

输入：JSON 数组或 JSONL。每条记录应至少包含：
sessionId、messageId、conversations、success、outputTokens。

运行示例：
    python sample_pipeline.py --input samples.jsonl --output result.jsonl

启用 Model API 语义评价（接口按 OpenAI Chat Completions 兼容格式）：
    # 方式一：环境变量
    $env:MODEL_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    $env:MODEL_API_KEY = "sk-xxx"
    $env:MODEL_NAME = "qwen3.5-plus"

    # 方式二：.env 文件（推荐，已在 .gitignore 中）
    # .env 内容见 .env.example

    python sample_pipeline.py --input samples.jsonl --output result.jsonl --model-eval

脚本保持单文件，便于先用样本集跑通链路；后续再拆分算子、替换规则或接入调度框架。
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
from typing import Any, Iterable


ASSISTANT_ROLES = {"assistant", "bot", "ai", "model"}
USER_ROLES = {"user", "human"}
NEGATIVE_THUMBS = {"down", "negative", "bad", "dislike", "0", "false"}
POSITIVE_THUMBS = {"up", "positive", "good", "like", "1", "true"}


def load_dotenv(path: str | None = None) -> None:
    """从 .env 文件加载环境变量（极简版，不依赖 python-dotenv）。

    优先级：已设的系统环境变量 > .env 文件。
    """
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
    """对结构化数据做稳定哈希，用于去重和生成样本 ID。"""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON 文件的顶层必须是数组")
        records = data
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    invalid = [type(record).__name__ for record in records if not isinstance(record, dict)]
    if invalid:
        raise ValueError(f"输入中存在非对象记录：{invalid[:3]}")
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def redact_text(value: str) -> str:
    """脱敏：身份证号、手机号、银行卡号、token/key。

    客户要求：身份证、手机号、银行卡号、病历明细、地址等敏感字段
    全部替换为占位符，保留业务语义，不保留原始隐私值。
    """
    # 手机号
    value = re.sub(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)", "[PHONE]", value)
    # 身份证号（15位或18位）
    value = re.sub(r"(?<!\d)\d{15}(?!\d)", "[ID_CARD]", value)
    value = re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "[ID_CARD]", value)
    # 银行卡号（16-19位纯数字）
    value = re.sub(r"(?<!\d)\d{16,19}(?!\d)", "[BANK_CARD]", value)
    # 邮箱
    value = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", value)
    # Bearer token / API key
    value = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[TOKEN]", value)
    value = re.sub(r"(?i)(api[-_ ]?key\s*[:=]\s*)[^\s,;]+", r"\1[TOKEN]", value)
    return value


def normalize_conversations(value: Any, redact: bool) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    conversations: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("from", item.get("role", ""))).strip().lower()
        content = item.get("value", item.get("content", ""))
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if redact:
            content = redact_text(content)
        if role in ASSISTANT_ROLES:
            role = "assistant"
        elif role in USER_ROLES:
            role = "user"
        if role and content.strip():
            conversations.append({"from": role, "value": content.strip()})
    return conversations


def first_value(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if record.get(name) not in (None, ""):
            return record[name]
    return None


def prepare_dataset(raw_records: list[dict[str, Any]], redact: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """算子 1：字段规整、去重、无效记录过滤和基础脱敏。

    只过滤结构无效（缺 ID、conversations 解析失败、重复）；
    业务状态异常（success=false、空 assistant 回答）通过 _prepareWarnings 标记，
    交给后续算子处理——难例挖掘需要看到失败样本。
    """
    prepared: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for index, original in enumerate(raw_records):
        record = copy.deepcopy(original)
        session_id = first_value(record, "sessionId", "sessionID")
        message_id = first_value(record, "messageId", "messageID")
        conversations = normalize_conversations(record.get("conversations"), redact)

        # --- 结构校验：这些是真正该丢的 ---
        reject_reasons: list[str] = []
        if not session_id:
            reject_reasons.append("missing_sessionId")
        if not message_id:
            reject_reasons.append("missing_messageId")
        if not conversations:
            reject_reasons.append("empty_conversations")

        dedup_key = f"{session_id}:{message_id}" if session_id and message_id else stable_hash(conversations)
        if dedup_key in seen_keys:
            reject_reasons.append("duplicate")

        if reject_reasons:
            rejected.append({"inputIndex": index, "reasons": reject_reasons, "messageId": message_id})
            continue

        seen_keys.add(dedup_key)
        record["sessionId"] = str(session_id)
        record["messageId"] = str(message_id)
        record["conversations"] = conversations
        record["sampleId"] = stable_hash({"sessionId": session_id, "messageId": message_id})[:20]

        # --- 业务异常标记：不丢，只记录 warnings 给后续算子参考 ---
        warnings: list[str] = []
        has_assistant = any(message["from"] == "assistant" for message in conversations)
        if record.get("success") is False:
            warnings.append("generation_failed")
        if not has_assistant:
            warnings.append("missing_assistant_answer")
        if warnings:
            record["_prepareWarnings"] = warnings

        # 这些字段不参与五个算子的样本判断，不进入输出样本。
        record.pop("traceId", None)
        record.pop("userId", None)

        if isinstance(record.get("userFeedback"), dict) and redact:
            feedback = record["userFeedback"]
            if isinstance(feedback.get("comment"), str):
                feedback["comment"] = redact_text(feedback["comment"])
        prepared.append(record)

    return prepared, rejected


def extract_feedback(record: dict[str, Any]) -> tuple[str, str | None]:
    feedback = record.get("userFeedback")
    if not isinstance(feedback, dict):
        return "unknown", None
    thumbs = str(feedback.get("thumbs", "")).strip().lower()
    if thumbs in POSITIVE_THUMBS:
        return "positive", thumbs
    if thumbs in NEGATIVE_THUMBS:
        return "negative", thumbs
    return "unknown", thumbs or None


def event_names(record: dict[str, Any]) -> set[str]:
    events = record.get("userInteractionEvents")
    if not isinstance(events, list):
        return set()
    return {str(event).strip().lower() for event in events if event is not None}


def numeric(record: dict[str, Any], name: str) -> float | None:
    value = record.get(name)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def assistant_text(record: dict[str, Any]) -> str:
    return "\n".join(
        message["value"]
        for message in record.get("conversations", [])
        if message.get("from") == "assistant"
    ).strip()


class ModelApiClient:
    """极简 OpenAI-compatible Model API 客户端。"""

    def __init__(self, url: str, api_key: str | None, model: str, timeout: int = 60):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def evaluate(self, record: dict[str, Any]) -> dict[str, Any]:
        prompt = {
            "task": "根据用户问题和助手回答，评价样本是否值得进入高质量数据候选集。只返回 JSON。",
            "output_schema": {
                "task_satisfaction": "0到1之间的小数，回答是否满足用户任务",
                "answer_quality": "0到1之间的小数，内容正确性、完整性和可用性综合分",
                "complexity": "0到1之间的小数，任务约束和推理复杂度",
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
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            content = response_body["choices"][0]["message"]["content"]
            result = parse_json_object(content)
            return {
                "status": "ok",
                "taskSatisfaction": clamp_score(result.get("task_satisfaction")),
                "answerQuality": clamp_score(result.get("answer_quality")),
                "complexity": clamp_score(result.get("complexity")),
                "reason": str(result.get("reason", ""))[:200],
            }
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError, TypeError) as exc:
            return {"status": "error", "reason": f"model_api_error: {exc}"}


def parse_json_object(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    text = str(content).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response is not a JSON object")
    result = json.loads(text[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("model response JSON is not an object")
    return result


def clamp_score(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


# --- 阈值配置（方案确认后的值） ---
THRESHOLD_SLOW_REQUEST_MS = 15_000       # duration > 15s 算慢请求
THRESHOLD_SLOW_TTFT_MS = 5_000           # ttft > 5s 算慢首 token
SEMANTIC_QUALIFIED_MIN = 0.6             # 语义评分 >= 0.6 可判合格
SEMANTIC_UNQUALIFIED_MAX = 0.4           # 语义评分 < 0.4 判不合格


def analyze_feedback(record: dict[str, Any], model_client: ModelApiClient | None) -> dict[str, Any]:
    """算子 3：反馈、交互和性能指标分析。"""
    feedback_status, thumbs = extract_feedback(record)
    events = event_names(record)
    duration = numeric(record, "durationMs")
    ttft = numeric(record, "ttftMs")
    output_tokens = numeric(record, "outputTokens")
    analysis = {
        "feedbackStatus": feedback_status,
        "feedbackThumbs": thumbs,
        "interactionSignals": sorted(events),
        "performanceFlags": [],
        "semanticEvaluation": {"status": "not_run"},
    }
    if duration is not None and duration > THRESHOLD_SLOW_REQUEST_MS:
        analysis["performanceFlags"].append("slow_request")
    if ttft is not None and ttft > THRESHOLD_SLOW_TTFT_MS:
        analysis["performanceFlags"].append("slow_first_token")
    if output_tokens is not None and output_tokens <= 0:
        analysis["performanceFlags"].append("empty_output")

    if model_client is not None and record.get("success") is True and assistant_text(record):
        analysis["semanticEvaluation"] = model_client.evaluate(record)
    return analysis


def find_hard_case(record: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    """算子 2：规则优先的难例挖掘。

    触发条件：请求失败 / 负反馈 / 用户停止生成 / regenerate / 空输出 / 语义评分低。
    """
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
    semantic = analysis.get("semanticEvaluation", {})
    if semantic.get("status") == "ok":
        if semantic.get("taskSatisfaction") is not None and semantic["taskSatisfaction"] < SEMANTIC_QUALIFIED_MIN:
            reasons.append("low_task_satisfaction")
        if semantic.get("answerQuality") is not None and semantic["answerQuality"] < SEMANTIC_QUALIFIED_MIN:
            reasons.append("low_answer_quality")
    return {"isHardExample": bool(reasons), "reasons": reasons}


def classify_candidate(record: dict[str, Any], analysis: dict[str, Any], hard_case: dict[str, Any]) -> dict[str, Any]:
    """算子 4：确定性候选样本分流。

    规则：
    - request_not_successful / empty_assistant_answer / empty_output → 直接 unqualified
    - negative_feedback → 进 pending（负反馈可能是回答方向问题而非质量问题）
    - 语义评分：>=0.6 合格 / <0.4 不合格 / 中间区间 pending
    """
    reject_reasons: list[str] = []
    pending_reasons: list[str] = []

    if record.get("success") is not True:
        reject_reasons.append("request_not_successful")
    if not assistant_text(record):
        reject_reasons.append("empty_assistant_answer")
    output_tokens = numeric(record, "outputTokens")
    if output_tokens is not None and output_tokens <= 0:
        reject_reasons.append("empty_output")

    # 负反馈 → pending，不是 reject
    if analysis["feedbackStatus"] == "negative":
        pending_reasons.append("negative_feedback")

    # 硬拒绝优先级最高
    if reject_reasons:
        return {"filterResult": "unqualified", "filterReasons": reject_reasons}

    semantic = analysis.get("semanticEvaluation", {})
    if semantic.get("status") == "ok":
        task_score = semantic.get("taskSatisfaction")
        answer_score = semantic.get("answerQuality")
        qualified = (
            task_score is not None
            and answer_score is not None
            and task_score >= SEMANTIC_QUALIFIED_MIN
            and answer_score >= SEMANTIC_QUALIFIED_MIN
        )
        unqualified = (
            task_score is not None
            and answer_score is not None
            and (task_score < SEMANTIC_UNQUALIFIED_MAX or answer_score < SEMANTIC_UNQUALIFIED_MAX)
        )

        if qualified and not pending_reasons:
            return {"filterResult": "qualified", "filterReasons": ["technical_checks_passed", "semantic_scores_passed"]}
        if unqualified:
            return {"filterResult": "unqualified", "filterReasons": ["semantic_scores_below_threshold"] + pending_reasons}
        # 语义评分在中间区间 或 有待处理的 pending 原因
        return {"filterResult": "pending", "filterReasons": ["semantic_score_in_middle_or_missing"] + pending_reasons}

    # 无语义评分
    if pending_reasons:
        return {"filterResult": "pending", "filterReasons": pending_reasons}
    if hard_case["isHardExample"] and analysis["feedbackStatus"] == "unknown":
        return {"filterResult": "pending", "filterReasons": ["hard_signal_needs_review"]}
    return {"filterResult": "qualified", "filterReasons": ["technical_checks_passed"]}


def scene_key(record: dict[str, Any]) -> str:
    """场景键：用于算子 5 统计高频/低频场景。

    systemName + category + X-Ai-Scene-Name 组合唯一标识一个业务场景。
    """
    return "|".join(
        str(record.get(field) or "unknown")
        for field in ("systemName", "category", "X-Ai-Scene-Name")
    )


def select_high_value(
    record: dict[str, Any],
    analysis: dict[str, Any],
    candidate: dict[str, Any],
    scene_counts: Counter[str],
    rare_threshold: int,
) -> dict[str, Any]:
    """算子 5：从合格样本中选增量高价值样本。"""
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

    semantic = analysis.get("semanticEvaluation", {})
    if semantic.get("complexity") is not None and semantic["complexity"] >= 0.7:
        reasons.append("complex_task")
    if scene_counts[scene_key(record)] <= rare_threshold:
        reasons.append("rare_production_scene")

    return {"isHighValue": bool(reasons), "reasons": reasons}


def run_pipeline(
    raw_records: list[dict[str, Any]],
    redact: bool,
    model_client: ModelApiClient | None,
    rare_threshold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared, rejected = prepare_dataset(raw_records, redact=redact)
    staged: list[dict[str, Any]] = []
    for record in prepared:
        analysis = analyze_feedback(record, model_client)
        hard_case = find_hard_case(record, analysis)
        candidate = classify_candidate(record, analysis, hard_case)
        staged.append({"record": record, "analysis": analysis, "hardCase": hard_case, "candidate": candidate})

    qualified_counts = Counter(
        scene_key(item["record"]) for item in staged if item["candidate"]["filterResult"] == "qualified"
    )
    output: list[dict[str, Any]] = []
    for item in staged:
        record = item["record"]
        high_value = select_high_value(record, item["analysis"], item["candidate"], qualified_counts, rare_threshold)
        result = copy.deepcopy(record)
        # 清理 _prepareWarnings（调试用），保留 pipeline 结果
        record.pop("_prepareWarnings", None)
        result["pipeline"] = {
            "feedbackAnalysis": item["analysis"],
            "hardCase": item["hardCase"],
            "candidate": item["candidate"],
            "highValue": high_value,
        }
        output.append(result)
    return output, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="数据飞轮五算子脚本")
    parser.add_argument("--input", required=True, help="输入 JSON 数组或 JSONL 文件")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件")
    parser.add_argument("--rejected-output", help="可选：输出规整阶段被过滤的记录及原因")
    parser.add_argument("--model-eval", action="store_true", help="调用 Model API 做语义评价")
    parser.add_argument("--model-api-url", default=os.getenv("MODEL_API_URL"), help="OpenAI-compatible chat completions URL")
    parser.add_argument("--model-api-key", default=os.getenv("MODEL_API_KEY"), help="Model API Key")
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "qwen3.6"), help="模型名")
    parser.add_argument("--model-timeout", type=int, default=120, help="单条模型调用超时秒数（dashscope 高峰可能 60-90s）")
    parser.add_argument("--rare-threshold", type=int, default=3, help="同场景合格样本数 <= 该值时标记为低频场景（方案确认值：3）")
    parser.add_argument("--no-redact", action="store_true", help="关闭脱敏，仅建议用于调试")
    return parser.parse_args()


def main() -> int:
    load_dotenv()  # 加载 .env 文件（优先级低于已设环境变量）

    args = parse_args()
    if args.model_eval and not args.model_api_url:
        print("--model-eval 需要 --model-api-url 或 MODEL_API_URL（或 .env 文件）", file=sys.stderr)
        return 2

    model_client = None
    if args.model_eval:
        model_client = ModelApiClient(args.model_api_url, args.model_api_key, args.model_name, args.model_timeout)

    raw_records = load_records(Path(args.input))
    output, rejected = run_pipeline(
        raw_records,
        redact=not args.no_redact,
        model_client=model_client,
        rare_threshold=max(args.rare_threshold, 0),
    )
    write_jsonl(Path(args.output), output)
    if args.rejected_output:
        write_jsonl(Path(args.rejected_output), rejected)

    counts = Counter(item["pipeline"]["candidate"]["filterResult"] for item in output)
    high_value_count = sum(1 for item in output if item["pipeline"]["highValue"]["isHighValue"])
    print(json.dumps({
        "input": len(raw_records),
        "prepared": len(output),
        "rejectedInPreparation": len(rejected),
        "candidateCounts": dict(counts),
        "highValue": high_value_count,
        "output": args.output,
    }, ensure_ascii=False))
    if rejected:
        print("规整阶段被过滤的记录（仅结构无效）：", json.dumps(rejected[:10], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
