#!/usr/bin/env python3
"""
Relay • Feed-Style Summariser (CLI) – File‑based version

* Amazon Bedrock for inference
* LaunchDarkly AI Configs for prompt/model selection
* Asks for a path to a .txt file, reads the **entire** file into the {{DOCUMENT}} placeholder.
* Provides **neutral defaults** for the prompt variables `audience`, `brand_voice`, and `cta`,
  which can be overridden via environment variables **or** edited directly in code.
"""

import os, sys, time, random, logging, dotenv, boto3
from collections import defaultdict
from typing import Dict, Any, List, Tuple
from botocore.exceptions import ClientError

# LaunchDarkly
import ldclient
from ldclient.config import Config
from ldclient.context import Context
from ldai.client import LDAIClient, AIConfig, ModelConfig, LDMessage, ProviderConfig
from ldai.tracker import FeedbackKind


# ───────────────────────── logging ───────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
)
log = logging.getLogger("relay-feed-cli")


# ─────────────────── helper: extract model params ────────────────────

def extract_params(model_cfg) -> Dict[str, Any]:
    if hasattr(model_cfg, "_parameters") and isinstance(model_cfg._parameters, dict):
        return model_cfg._parameters
    if hasattr(model_cfg, "parameters") and isinstance(model_cfg.parameters, dict):
        return model_cfg.parameters
    params = {}
    for key in ("temperature", "top_p", "topP", "max_tokens", "maxTokens"):
        if hasattr(model_cfg, key):
            params[key] = getattr(model_cfg, key)
    return params


# ─────────────────────── LaunchDarkly wrapper ────────────────────────
class LDClient:
    def __init__(self, sdk_key: str, ai_config_id: str):
        ldclient.set_config(Config(sdk_key))
        self._ld = ldclient.get()
        self._ai = LDAIClient(self._ld)
        self._config_id = ai_config_id

    def get_config(self, ctx: Context, variables: Dict[str, Any]):
        try:
            fb = self._fallback()
            return self._ai.config(self._config_id, ctx, fb, variables)
        except Exception as exc:
            log.warning("LaunchDarkly unavailable – using fallback (%s)", exc)
            return self._fallback(), None

    @staticmethod
    def _fallback():
        return AIConfig(
            enabled=True,
            provider=ProviderConfig(name="bedrock"),
            model=ModelConfig(
                name="anthropic.claude-v2:1",
                parameters=dict(temperature=0.4, top_p=0.9, max_tokens=800),
            ),
            messages=[
                LDMessage(
                    role="system",
                    content="You turn long text into 1-3 feed-style bullets (≤ 280 chars each).",
                )
            ],
        )


# ───────────────────────── Bedrock wrapper ───────────────────────────
class Bedrock:
    def __init__(self, region: str):
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def stream(self, model_id: str, system_prompt: str,
               messages: List[Dict[str, Any]], params: Dict[str, Any]):
        req = dict(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=messages,
            inferenceConfig=params,
        )
        return self._client.converse_stream(**req)["stream"]

    def parse(self, stream, tracker):
        full, first_token = [], None
        metric = {"$metadata": {"httpStatusCode": 200}}
        start = time.time()

        for ev in stream:
            if "contentBlockDelta" in ev:
                chunk = ev["contentBlockDelta"]["delta"]["text"]
                if first_token is None:
                    first_token = (time.time() - start) * 1000
                    metric.setdefault("metrics", {})["timeToFirstToken"] = first_token
                print(chunk, end="", flush=True)
                full.append(chunk)
            if "metadata" in ev:
                md = ev["metadata"]
                if "usage" in md:
                    metric["usage"] = md["usage"]
                if "metrics" in md:
                    metric.setdefault("metrics", {}).update(md["metrics"])

        print()  # newline
        if tracker:
            tracker.track_bedrock_converse_metrics(metric)
            tracker.track_success()
            if first_token:
                tracker.track_time_to_first_token(first_token)
        return "".join(full), metric


# ─────────── convert LaunchDarkly messages → Bedrock format ──────────

def to_bedrock(cfg: AIConfig, doc_text: str, audience: str,
               brand_voice: str, cta: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Replace placeholders in LD messages and ensure the document is included."""
    system_prompt = ""
    messages = []

    for i, m in enumerate(cfg.messages):
        if i == 0 and m.role == "system":
            system_prompt = m.content
            continue
        role = "user" if m.role == "user" else "assistant"
        content = (
            m.content
            .replace("{{document_chunk}}", doc_text)
            .replace("{{audience}}", audience)
            .replace("{{brand_voice}}", brand_voice)
            .replace("{{cta}}", cta)
        )
        messages.append({"role": role, "content": [{"text": content}]})

    # Ensure the document itself is part of the conversation at least once
    if not any(msg["content"][0]["text"] == doc_text for msg in messages):
        messages.append({"role": "user", "content": [{"text": doc_text}]})

    return system_prompt, messages


# ───────────────────────────── main ──────────────────────────────────

def main():
    dotenv.load_dotenv()
    if not os.getenv("LD_SERVER_KEY"):
        log.error("LD_SERVER_KEY missing – add it to .env")
        sys.exit(1)

    # ── neutral prompt defaults (override via ENV) ───────────────────
    DEFAULT_AUDIENCE = os.getenv("AUDIENCE", "reader")
    DEFAULT_BRAND_VOICE = os.getenv("BRAND_VOICE", "Neutral")
    DEFAULT_CTA = os.getenv("CTA", "Learn more")

    # ── run‑seed (fresh per script launch) ────────────────────────────
    SEED = random.randint(1, 10)
    log.info("Run‑seed: %s", SEED)

    ld = LDClient(os.getenv("LD_SERVER_KEY"),
                  os.getenv("LD_AI_CONFIG_ID", "relay-feed"))
    bedrock = Bedrock(os.getenv("AWS_REGION", "us-east-1"))

    ctx = (
        Context.builder("cli-user")
        .set("source", "cli")
        .set("seed", SEED)
        .build()
    )

    # ── initial AI Config just to reveal the model before first input ─
    init_vars = {
        "document_chunk": "",
        "audience": DEFAULT_AUDIENCE,
        "brand_voice": DEFAULT_BRAND_VOICE,
        "cta": DEFAULT_CTA,
        "seed": SEED,
    }
    cfg_init, _ = ld.get_config(ctx, init_vars)
    init_params = extract_params(cfg_init.model)
    print(f"\n🔎  Model for this session: {cfg_init.model.name}")
    if init_params:
        print(f"     with params: {init_params}\n")

    metrics = defaultdict(list)

    print("🤖  I'm happy to help! Provide a path to a .txt file for summarisation below.")
    print("(type 'exit' to quit)\n")

    while True:
        # ── get path to txt file ─────────────────────────────────────
        try:
            path = input("Enter path to .txt file: ").strip()
        except (EOFError, KeyboardInterrupt):
            path = "exit"
        if not path:
            continue
        if path.lower() == "exit":
            print("Goodbye!")
            finish(metrics)
            return
        if not os.path.isfile(path):
            print("❌  File not found. Please provide a valid path.\n")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = f.read().strip()
        except Exception as exc:
            print(f"❌  Could not read file: {exc}\n")
            continue
        if not doc:
            print("⚠️  File is empty. Please provide a file with content.\n")
            continue

        audience = os.getenv("AUDIENCE", DEFAULT_AUDIENCE)
        brand_voice = os.getenv("BRAND_VOICE", DEFAULT_BRAND_VOICE)
        cta = os.getenv("CTA", DEFAULT_CTA)

        variables = {
            "document_chunk": doc,
            "audience": audience,
            "brand_voice": brand_voice,
            "cta": cta,
            "seed": SEED,
        }
        cfg, tracker = ld.get_config(ctx, variables)

        params = extract_params(cfg.model)
        inf_cfg = {
            "temperature": params.get("temperature"),
            "topP": params.get("top_p") or params.get("topP"),
            "maxTokens": params.get("max_tokens") or params.get("maxTokens"),
        }
        inf_cfg = {k: v for k, v in inf_cfg.items() if v is not None}

        system_prompt, msgs = to_bedrock(cfg, doc, audience, brand_voice, cta)

        print("\n--- Generating feed messages ---\n")
        try:
            stream = bedrock.stream(cfg.model.name, system_prompt, msgs, inf_cfg)
            reply, metric = bedrock.parse(stream, tracker)
            for k, v in metric.get("usage", {}).items():
                metrics[k].append(v)
            for k, v in metric.get("metrics", {}).items():
                metrics[k].append(v)
        except ClientError as err:
            log.error("AWS error: %s", err)
            continue
        except Exception as exc:
            log.exception("Unexpected error: %s", exc)
            continue

        fb = input("\n👍 Was this helpful? (y/n) ").strip().lower()
        if fb.startswith("y") and tracker:
            tracker.track_feedback({"kind": FeedbackKind.Positive})
        elif fb.startswith("n") and tracker:
            tracker.track_feedback({"kind": FeedbackKind.Negative})
        if tracker:
            ldclient.get().flush()

        print("\nProvide another file, or press Ctrl‑C / type 'exit' to quit.\n")


def finish(metrics):
    if not metrics:
        return
    print("\nSession metrics")
    print("----------------")
    for k, v in metrics.items():
        avg = sum(v) / len(v)
        print(f"{k:>18}: {avg:.1f}   (n={len(v)})")


if __name__ == "__main__":
    main()
