"""
Use an LLM to label truth social posts into pre-defined buckets.

Json output is written to a folder under ``out/truth_buckets`` 

intermediate output is stored so can re-run without re-doing all.
"""


# export OPENAI_API_KEY=
# python -m src.llm_pipeline.truth_social_llm_buckets --start 2025-07-01 --end 2025-07-02

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional
from urllib.parse import urlparse

import pandas as pd

# Allow running via `python -m src...` without manual PYTHONPATH tweaks.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import paths


#------------------------------------
# prompt and examples 
#------------------------------------


SYSTEM_PROMPT = "You label short social-media posts into one primary category from a fixed set. Be concise and deterministic."

BUCKET_DEFINITIONS = {
    "media_combat": "Media/Enemy-combat — attacking media, journalists, ‘fake news,’ or platforms.",
    "rally_promo": "Rally/Promotion — promoting rallies, livestreams, campaign events, or merch.",
    "courts_trials": "Law and order; Federal troop deployments — cases, judges, prosecutors, legal processes, or federal law-and-order crackdowns (e.g., National Guard deployments).",
    "elections_fraud": "Elections/Fraud — election integrity, fraud allegations, voting rules, or polling.",
    "immigration_border": "Immigration/Border — migrants, wall, asylum, or border security.",
    "economy_inflation": "Economy; Inflation; Trade — prices, jobs, taxes, markets, or trade.",
    "other": "Other/None — condolences, generic thanks, personal posts, or items without text (media-only).",
}


def _example_output(
    post_id: str,
    primary: str,
    *,
    secondary: Optional[str] = None,
    confidence: float = 0.9,
    is_media_only: bool = False,
    media_notes: str = "",
    rationale: str,
) -> Dict[str, object]:
    return {
        "post_id": post_id,
        "primary_bucket": primary,
        "secondary_bucket": secondary,
        "confidence": confidence,
        "is_media_only": is_media_only,
        "media_notes": media_notes,
        "rationale": rationale,
    }


FEW_SHOT_EXAMPLES = [
    {
        "input": {
            "post_id": "media_example",
            "post_text": "The Fake News New York Times refuses to cover our victory tonight.",
            "has_image": False,
            "has_video": False,
            "has_link": False,
            "media_domain": None,
        },
        "output": _example_output(
            "media_example",
            "media_combat",
            secondary=None,
            rationale="Attacks hostile media coverage",
        ),
    },
    {
        "input": {
            "post_id": "rally_example",
            "post_text": "Join me LIVE in Michigan tonight at 7pm! RSVP now.",
            "has_image": True,
            "has_video": False,
            "has_link": True,
            "media_domain": "truthsocial.com",
        },
        "output": _example_output(
            "rally_example",
            "rally_promo",
            secondary=None,
            rationale="Invites followers to campaign rally",
        ),
    },
    {
        "input": {
            "post_id": "courts_example",
            "post_text": "Corrupt Judge Smith just delayed our case again. Total election interference!",
            "has_image": False,
            "has_video": False,
            "has_link": False,
            "media_domain": None,
        },
        "output": _example_output(
            "courts_example",
            "courts_trials",
            secondary="elections_fraud",
            rationale="Discusses judge and litigation timeline",
        ),
    },
    {
        "input": {
            "post_id": "law_order_example",
            "post_text": "If the crime wave continues, I will send the National Guard into Chicago immediately!",
            "has_image": False,
            "has_video": False,
            "has_link": False,
            "media_domain": None,
        },
        "output": _example_output(
            "law_order_example",
            "courts_trials",
            secondary=None,
            rationale="Promises federal law-and-order intervention",
        ),
    },
    {
        "input": {
            "post_id": "elections_example",
            "post_text": "Democrats are trying to rig 2024 with illegal drop boxes—watch the polls!",
            "has_image": False,
            "has_video": False,
            "has_link": False,
            "media_domain": None,
        },
        "output": _example_output(
            "elections_example",
            "elections_fraud",
            secondary=None,
            rationale="Claims election is being rigged",
        ),
    },
    {
        "input": {
            "post_id": "immigration_example",
            "post_text": "Finish the wall and deport the violent criminals flooding across our border!",
            "has_image": False,
            "has_video": False,
            "has_link": False,
            "media_domain": None,
        },
        "output": _example_output(
            "immigration_example",
            "immigration_border",
            secondary=None,
            rationale="Demands tougher border security",
        ),
    },
    {
        "input": {
            "post_id": "economy_example",
            "post_text": "Crooked Biden gave you inflation. Under TRUMP we had jobs, low prices, and big paychecks!",
            "has_image": False,
            "has_video": False,
            "has_link": False,
            "media_domain": None,
        },
        "output": _example_output(
            "economy_example",
            "economy_inflation",
            secondary=None,
            rationale="Focuses on inflation and jobs",
        ),
    },
    {
        "input": {
            "post_id": "ballroom_example",
            "post_text": "The Mar-a-Lago ballroom renovation is stunning—wait until you see the chandeliers tonight!",
            "has_image": True,
            "has_video": False,
            "has_link": False,
            "media_domain": "truthsocial.com",
        },
        "output": _example_output(
            "ballroom_example",
            "other",
            secondary=None,
            rationale="Updates ceremonial venue décor",
        ),
    },
    {
        "input": {
            "post_id": "media_only_example",
            "post_text": "",
            "has_image": True,
            "has_video": False,
            "has_link": False,
            "media_domain": "rumble.com",
        },
        "output": _example_output(
            "media_only_example",
            "other",
            secondary=None,
            is_media_only=True,
            media_notes="Photo-only post",
            rationale="No accompanying text provided",
        ),
    },
]


#------------------------------------
# prep truth social posts
#------------------------------------



def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_html_content(value: object) -> str:
    """Normalize Mastodon/Truth HTML content into plaintext."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = re.sub(r"(?i)<\s*br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</\s*p\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\r?\n\s*\n+", "\n\n", text)
    return text.strip()


def detect_media_flags(media_value: object) -> Dict[str, Optional[object]]:
    """Return heuristics about attached media."""

    if media_value is None or (isinstance(media_value, float) and pd.isna(media_value)):
        return {
            "has_image": False,
            "has_video": False,
            "media_domain": None,
        }

    media_str = str(media_value).strip()
    if not media_str:
        return {
            "has_image": False,
            "has_video": False,
            "media_domain": None,
        }

    lowered = media_str.lower()
    has_image = any(ext in lowered for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"))
    has_video = any(ext in lowered for ext in (".mp4", ".mov", ".m4v", ".webm"))

    try:
        domain = urlparse(media_str).netloc or None
    except Exception:
        domain = None

    return {
        "has_image": has_image,
        "has_video": has_video,
        "media_domain": domain,
    }


def detect_has_link(text: str) -> bool:
    return bool(re.search(r"https?://", text))


@dataclass
class PostForLabeling:
    post_id: str
    text: str
    created_at: pd.Timestamp
    has_image: bool
    has_video: bool
    has_link: bool
    media_domain: Optional[str]

    @property
    def is_media_only(self) -> bool:
        return not self.text.strip()


def load_posts(
    csv_path: Path,
    start_date: str,
    end_date: str,
    timezone: str = "UTC",
) -> List[PostForLabeling]:
    df = pd.read_csv(csv_path, parse_dates=["created_at"])
    text_col = "content" if "content" in df.columns else "text"

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df = df.sort_values("created_at")

    start_ts = pd.Timestamp(start_date, tz=timezone)
    end_ts = pd.Timestamp(end_date, tz=timezone) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    start_ts_utc = start_ts.tz_convert("UTC")
    end_ts_utc = end_ts.tz_convert("UTC")

    mask = (df["created_at"] >= start_ts_utc) & (df["created_at"] <= end_ts_utc)
    subset = df.loc[mask].copy()

    posts: List[PostForLabeling] = []
    for _, row in subset.iterrows():
        post_id = str(row.get("id") or row.name)
        text = clean_html_content(row.get(text_col))
        media_info = detect_media_flags(row.get("media"))
        has_link = detect_has_link(text)
        posts.append(
            PostForLabeling(
                post_id=post_id,
                text=text,
                created_at=row["created_at"],
                has_image=bool(media_info["has_image"]),
                has_video=bool(media_info["has_video"]),
                has_link=has_link,
                media_domain=media_info["media_domain"],
            )
        )

    return posts


#------------------------------------
# prompt
#------------------------------------
def render_bucket_guidelines() -> str:
    lines = ["Bucket inventory:"]
    for key, description in BUCKET_DEFINITIONS.items():
        lines.append(f"- {key}: {description}")

    decision_rules = [
        "Decision rules:",
        "1. Pick exactly one primary_bucket from the list.",
        "2. Add a secondary_bucket only when a strong secondary theme is explicit; otherwise use null.",
        "3. If the post text is empty, set is_media_only = true and primary_bucket = other.",
        "4. When a post promotes a rally and attacks media, make rally_promo the primary bucket and media_combat the secondary bucket.",
        "5. When a court-related post mentions the media but focuses on the case, set primary_bucket = courts_trials.",
        "6. Posts about federal troop deployments, National Guard call-ups, or similar crackdowns belong in courts_trials.",
        "7. Ceremony or décor updates (ballrooms, receptions, etc.) belong in other unless another theme dominates.",
        "8. Lower the confidence when unsure instead of inventing new labels.",
        "9. Keep rationale under 15 words.",
    ]

    schema = {
        "post_id": "<string>",
        "primary_bucket": "<enum>",
        "secondary_bucket": "<enum or null>",
        "confidence": 0.0,
        "is_media_only": False,
        "media_notes": "<optional short string>",
        "rationale": "<<=15 words>",
    }

    lines.extend(["", *decision_rules, "", "Output only JSON Lines matching this schema:"])
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append("Few-shot examples:")

    for example in FEW_SHOT_EXAMPLES:
        input_payload = example["input"].copy()
        lines.append("Input:")
        lines.append(json.dumps(input_payload, ensure_ascii=False))
        lines.append("Output:")
        lines.append(json.dumps(example["output"], ensure_ascii=False))
        lines.append("")

    return "\n".join(lines).strip()


GUIDELINES_BLOCK = render_bucket_guidelines()


def build_prompt(posts: List[PostForLabeling]) -> str:
    chunks = [GUIDELINES_BLOCK, "Now label the following posts."]
    chunks.append("Return one JSON object per line in the same order.")
    for idx, post in enumerate(posts, start=1):
        payload = {
            "post_id": post.post_id,
            "post_text": post.text,
            "has_image": post.has_image,
            "has_video": post.has_video,
            "has_link": post.has_link,
            "media_domain": post.media_domain,
        }
        chunks.append(f"Post {idx}:")
        chunks.append(json.dumps(payload, ensure_ascii=False))
    return "\n\n".join(chunks)



class LLMCaller:
    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required")

        self._mode = "chat"
        self._client = None
        self._caller = None

        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)

            def _call(client, model: str, messages: List[Dict[str, str]]) -> str:
                # Newer models (e.g., gpt-5-mini) reject the temperature parameter, so omit it.
                response = client.responses.create(
                    model=model,
                    input=messages,
                )
                return getattr(response, "output_text", None) or str(response)

            self._mode = "responses"
            self._client = client
            self._caller = _call
            return
        except Exception:
            pass

        import openai  # type: ignore

        openai.api_key = api_key

        def _call(_client, model: str, messages: List[Dict[str, str]]) -> str:
            # Same rationale as above: omit temperature for universal compatibility.
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
            )
            return completion["choices"][0]["message"]["content"]

        self._mode = "chat"
        self._client = None
        self._caller = _call

    @property
    def mode(self) -> str:
        return self._mode

    def __call__(self, model: str, messages: List[Dict[str, str]]) -> str:
        assert self._caller is not None
        return self._caller(self._client, model, messages)


def iter_chunks(sequence: List[PostForLabeling], size: int) -> Iterator[List[PostForLabeling]]:
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


def parse_llm_response(raw: str) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for line in raw.splitlines():
        line = line.strip().strip("`").rstrip(",")
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "post_id" not in obj or "primary_bucket" not in obj:
            continue
        results.append(obj)
    return results


def save_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize_timezone_for_path(name: str) -> str:
    cleaned = name.strip().lower().replace("/", "__").replace(":", "-").replace(" ", "_")
    return re.sub(r"[^a-z0-9_.-]", "_", cleaned) or "tz"


def existing_ids(path: Path) -> Dict[str, Dict[str, object]]:
    records: Dict[str, Dict[str, object]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("post_id"):
                records[str(obj["post_id"])] = obj
    return records


#------------------------------------
# run llm calls
#------------------------------------
def run_labeling(
    start: str,
    end: str,
    *,
    csv_path: Optional[Path] = None,
    model: str = "gpt-4o-mini",
    chunk_size: int = 5,
    timezone: str = "UTC",
    output_dir: Optional[Path] = None,
) -> None:
    csv_path = csv_path or paths.truth_archive_csv()
    base_output_dir = output_dir or (paths.output_subdir("truth_buckets", ensure=True) / "labels")
    output_dir = base_output_dir
    logs_dir = paths.output_subdir("truth_buckets", ensure=True) / "logs"

    ensure_directory(output_dir)
    ensure_directory(logs_dir)

    start_clean = start.replace("-", "")
    end_clean = end.replace("-", "")
    tz_slug = sanitize_timezone_for_path(timezone)
    daily_dir = output_dir / tz_slug
    ensure_directory(daily_dir)
    log_path = logs_dir / f"responses_{start_clean}_{end_clean}.log.jsonl"

    posts = load_posts(csv_path, start, end, timezone=timezone)
    if not posts:
        print("No posts found for requested window.")
        return

    daily_paths: Dict[str, Path] = {}
    post_lookup: Dict[str, PostForLabeling] = {}
    already: Dict[str, Dict[str, object]] = {}
    for post in posts:
        local_key = post.created_at.tz_convert(timezone).strftime("%Y%m%d")
        target_path = daily_dir / f"labels_{local_key}.jsonl"
        daily_paths.setdefault(local_key, target_path)
        post_lookup[post.post_id] = post

    for path in daily_paths.values():
        existing = existing_ids(path)
        if existing:
            already.update(existing)

    todo = [post for post in posts if post.post_id not in already]

    if not todo:
        print("No posts to label — everything already processed.")
        return

    llm = LLMCaller()

    with log_path.open("a", encoding="utf-8") as log_handle:
        for batch in iter_chunks(todo, chunk_size):
            prompt = build_prompt(batch)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            raw_response = None
            attempt = 0
            while True:
                attempt += 1
                try:
                    raw_response = llm(model, messages)
                    break
                except Exception as exc:
                    if attempt > 3:
                        raise
                    wait_for = 2 ** attempt
                    print(f"LLM call failed ({exc}); retrying in {wait_for}s", file=sys.stderr)
                    time.sleep(wait_for)

            log_entry = {
                "ts": time.time(),
                "mode": llm.mode,
                "model": model,
                "n_items": len(batch),
                "response": raw_response,
            }
            log_handle.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            log_handle.flush()

            parsed = parse_llm_response(raw_response or "")
            if not parsed:
                print("Warning: no parseable JSON objects returned; skipping batch.", file=sys.stderr)
                continue

            rows_by_path: Dict[Path, List[Dict[str, object]]] = {}
            skipped_unknown = 0
            for row in parsed:
                post_id = str(row.get("post_id"))
                if not post_id:
                    continue
                post = post_lookup.get(post_id)
                if not post:
                    skipped_unknown += 1
                    continue
                if post_id in already:
                    continue
                local_key = post.created_at.tz_convert(timezone).strftime("%Y%m%d")
                target_path = daily_paths.get(local_key)
                if not target_path:
                    skipped_unknown += 1
                    continue
                rows_by_path.setdefault(target_path, []).append(row)
                already[post_id] = row

            if skipped_unknown:
                print(f"Warning: skipped {skipped_unknown} rows without matching post metadata.", file=sys.stderr)

            for path, rows in rows_by_path.items():
                save_jsonl(path, rows)
                print(f"Processed {len(rows)} posts → {path}")


#------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label Truth Social posts into predefined buckets using an LLM.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional configuration YAML overriding config.yml")
    parser.add_argument("--csv", type=Path, default=None, help="Override input CSV path")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07", help="OpenAI model name (default: gpt-5-mini-2025-08-07)")
    parser.add_argument("--chunk-size", type=int, default=5, help="Posts per LLM request (default: 5)")
    parser.add_argument("--timezone", default="UTC", help="Timezone for filtering posts (default: UTC)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination root for per-day label JSONL files (default: out/truth_buckets/labels)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.config:
        paths.configure(args.config)

    run_labeling(
        args.start,
        args.end,
        csv_path=args.csv,
        model=args.model,
        chunk_size=args.chunk_size,
        timezone=args.timezone,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
