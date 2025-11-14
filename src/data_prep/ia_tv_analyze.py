#!/usr/bin/env python3


"""
Clean transcripts and count term occcurrences

choose --clean none to debug raw text, or --clean captions/html/auto to get comparable per-word densities across shows.


-- sample one day

python -m src.data_prep.ia_tv_analyze \
  --cache-dir data_transcripts \
  --start 2024-05-11 --end 2024-05-11 \
  --term "Epstein" \
  --per-show-out data_out/tv_transcripts/testper_show_2024-05-11.csv \
  --daily-out data_out/tv_transcripts/testdaily_2024-05-11.csv \
  -v


"""

import os, sys, csv, gzip, json, re
import datetime as dt
from typing import Iterator, Dict, Any, Optional

import click

# cleaning regex
_SRT_TIME_RX = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}")
_VTT_TIME_RX = re.compile(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}")
_TIME_LINE_RX = re.compile(r"^\s*\d{1,2}:\d{2}\s?(?:am|pm)\s*$", re.I)
_HTML_TAG_RX = re.compile(r"<[^>\n]+>")

_JUNK_CHROME_RX = re.compile(
    r"(icon|logo|Donate|Sign up|Log in|Search|About|Blog|Projects|Help|Contact|Jobs|Volunteer|Upload|Wayback Machine|Internet Archive)",
    re.I,
)

_STOP_MARKERS = {
    "IN COLLECTIONS", "SIMILAR ITEMS", "Borrow Program",
    "Search the Wayback Machine", "Advanced Search",
}

def _iter_json_records(root: str, start: dt.date, end: dt.date) -> Iterator[Dict[str, Any]]:
    cur = start; one = dt.timedelta(days=1)
    while cur <= end:
        ddir = os.path.join(root, cur.isoformat())
        if os.path.isdir(ddir):
            for fn in os.listdir(ddir):
                if not fn.endswith(".json.gz"): continue
                path = os.path.join(ddir, fn)
                try:
                    with gzip.open(path, "rt", encoding="utf-8") as f:
                        yield json.load(f)
                except Exception as e:
                    print(f"[warn] skip unreadable {path}: {e}", file=sys.stderr)
        cur += one

def _compile_term(term: str, word_boundary: bool):
    if word_boundary:
        pat = rf"\b{re.escape(term)}\b"
    else:
        pat = re.escape(term)
    return re.compile(pat, re.I)

#  cleaning utils 

def _normalize_caption_text(raw: str) -> str:
    lines, in_block = [], False
    for ln in raw.splitlines():
        s = ln.strip()
        if not s: continue
        if s.isdigit():  # SRT idx
            continue
        if "-->" in s and (_SRT_TIME_RX.search(s) or _VTT_TIME_RX.search(s)):
            continue
        if s.upper() == "WEBVTT":  # VTT header
            continue
        if s.startswith(("NOTE", "STYLE")):  # begin VTT block
            in_block = True
            continue
        if in_block:
            if s == "":
                in_block = False
            continue
        s = _HTML_TAG_RX.sub("", s)  # drop <c.colorXXX> etc.
        lines.append(s)
    txt = " ".join(lines)
    return re.sub(r"\s+", " ", txt).strip()

def _clean_htmlish_transcript(raw: str) -> str:
    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln]
    start_ix = None
    for i, ln in enumerate(lines):
        if _TIME_LINE_RX.match(ln):
            start_ix = i; break
    if start_ix is None:
        # fallback: just collapse whitespace; leave as-is
        return re.sub(r"\s+", " ", raw).strip()

    stop_ix = len(lines)
    for j in range(start_ix + 1, len(lines)):
        if any(lines[j].startswith(m) for m in _STOP_MARKERS):
            stop_ix = j; break

    core = lines[start_ix:stop_ix]
    core = [ln for ln in core if not _JUNK_CHROME_RX.search(ln)]
    core = [ln for ln in core if len(ln.split()) >= 3 or _TIME_LINE_RX.match(ln)]
    txt = " ".join(core)
    return re.sub(r"\s+", " ", txt).strip()

def _auto_clean(rec: Dict[str, Any]) -> str:
    txt = rec.get("transcript", "") or ""
    src = (rec.get("source") or "").lower()
    capf = (rec.get("caption_file") or "").lower()
    if src == "caption_file" or any(ext in capf for ext in (".srt", ".vtt", ".txt")):
        return _normalize_caption_text(txt)
    return _clean_htmlish_transcript(txt)

@click.command()
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False),
    default="data_transcripts",
    show_default=True,
    help="Directory created by src.data_prep.ia_tv_get",
)
@click.option("--start", type=click.DateTime(formats=["%Y-%m-%d"]), required=True)
@click.option("--end",   type=click.DateTime(formats=["%Y-%m-%d"]), required=True)
@click.option("--term", default="Epstein", show_default=True)
@click.option("--word-boundary/--no-word-boundary", default=True, show_default=True)
@click.option("--per-show-out", type=click.Path(dir_okay=False), default=None, help="CSV of per-program stats")
@click.option("--daily-out", type=click.Path(dir_okay=False), default="daily.csv", show_default=True, help="CSV of daily aggregates")
@click.option("--per-words", type=int, default=1000, show_default=True, help="density base (hits per N words)")
@click.option("--clean", type=click.Choice(["auto","none","captions","html"]), default="auto", show_default=True, help="cleaning applied before counting")
@click.option("-v", "--verbose", count=True)
def main(cache_dir, start, end, term, word_boundary, per_show_out, daily_out, per_words, clean, verbose):
    start_d, end_d = start.date(), end.date()
    if verbose:
        print(f"[info] analyzing {cache_dir} for {start_d}..{end_d} term={term!r} clean={clean}", file=sys.stderr)
    rx = _compile_term(term, word_boundary)

    per_show_rows = []
    daily: Dict[str, Dict[str, Any]] = {}

    for rec in _iter_json_records(cache_dir, start_d, end_d):
        day = rec.get("date","")
        raw_txt = rec.get("transcript","") or ""

        if clean == "none":
            used_txt = raw_txt
        elif clean == "captions":
            used_txt = _normalize_caption_text(raw_txt)
        elif clean == "html":
            used_txt = _clean_htmlish_transcript(raw_txt)
        else:  # auto
            used_txt = _auto_clean(rec)

        wc = len(used_txt.split())
        hits = len(rx.findall(used_txt))
        row = {
            "date": day,
            "identifier": rec.get("identifier",""),
            "title": rec.get("title",""),
            "channel": rec.get("channel",""),
            "hits": hits,
            "words": wc,
            "density_per_"+str(per_words): (hits / wc * per_words) if wc else 0.0,
        }
        per_show_rows.append(row)

        d = daily.setdefault(day, {"hits":0, "words":0, "shows":0, "shows_with_hits":0})
        d["hits"] += hits
        d["words"] += wc
        d["shows"] += 1
        if hits > 0: d["shows_with_hits"] += 1

    if per_show_out:
        with open(per_show_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            cols = ["date","identifier","title","channel","hits","words",f"density_per_{per_words}"]
            w.writerow(cols)
            for r in per_show_rows:
                w.writerow([r[c] for c in cols])
        if verbose:
            print(f"[write] per-show -> {per_show_out} ({len(per_show_rows)} rows)", file=sys.stderr)

    # fill zero days
    cur = start_d; one = dt.timedelta(days=1)
    while cur <= end_d:
        daily.setdefault(cur.isoformat(), {"hits":0, "words":0, "shows":0, "shows_with_hits":0})
        cur += one

    with open(daily_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = ["date","hits","words","shows","shows_with_hits",f"density_per_{per_words}"]
        w.writerow(cols)
        for day in sorted(daily.keys()):
            d = daily[day]
            density = (d["hits"]/d["words"]*per_words) if d["words"] else 0.0
            w.writerow([day, d["hits"], d["words"], d["shows"], d["shows_with_hits"], f"{density:.6f}"])
    if verbose:
        print(f"[write] daily -> {daily_out} ({len(daily)} rows)", file=sys.stderr)

if __name__ == "__main__":
    main()
