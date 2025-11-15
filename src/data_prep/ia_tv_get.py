#!/usr/bin/env python3

"""
Get data from internet archive - be sure to support them for their work!

In general it is better to use GDELT for this kind of analysis, but
either I couldn't get it to work correctly or they were missing the data for my sample.

--- test
python -m src.data_prep.ia_tv_get \
  --start 2024-05-11 --end 2024-05-11 \
  --chunk-days 1 --pause 1.1 --retries 5 --backoff 1.8 -vv \
  --out-dir data_transcripts



-- full window
python -m src.data_prep.ia_tv_get \
  --start 2024-10-01 --end 2025-10-21 \
  --chunk-days 7 --pause 1.1 --retries 5 --backoff 1.8 -v \
  --out-dir data_transcripts



-- for CNN use 
python -m src.data_prep.ia_tv_get \
  --start 2024-10-01 --end 2025-10-21 \
  --chunk-days 7 --pause 1.1 --retries 5 --backoff 1.8 \
  -n CNN \
  --out-dir data_transcripts_cnn -v


-- msnbc 
python -m src.data_prep.ia_tv_get \
  --start 2024-10-01 --end 2025-10-21 \
  -n MSNBCW \
  --out-dir data_transcripts_msnbc -v


"""

import os, sys, time, re, json, gzip
import datetime as dt
from typing import List, Dict, Tuple, Optional
from urllib.parse import quote

import click
import requests
from bs4 import BeautifulSoup
import hashlib


SCRAPE_HOSTS = [
    "https://archive.org/services/search/v1/scrape",
    "https://api.archive.org/search/v1/scrape",
]
META_URL    = "https://archive.org/metadata/{identifier}"
DL_URL      = "https://archive.org/download/{identifier}/{fname}"
DETAILS_URL = "https://archive.org/details/{identifier}"

UA = {"User-Agent": "ia-tv-cache/0.5 (+research)"}

_SRT_TIME_RX = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}")
_VTT_TIME_RX = re.compile(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}")
_TIME_LINE_RX = re.compile(r"^\s*\d{1,2}:\d{2}\s?(?:am|pm)\s*$", re.I)

def _normalize_caption_text(raw: str, ext_hint: str = "") -> str:
    """Strip SRT/VTT indices/timestamps → plain text (keeps punctuation)."""
    lines, in_block = [], False
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.isdigit():
            continue
        if "-->" in s and (_SRT_TIME_RX.search(s) or _VTT_TIME_RX.search(s)):
            continue
        if s.upper() == "WEBVTT":
            continue
        if s.startswith(("NOTE", "STYLE")):
            in_block = True
            continue
        if in_block:
            if s == "":
                in_block = False
            continue
        # very simple tag scrub; keep text
        s = re.sub(r"<[^>\n]>", "", s)
        lines.append(s)
    txt = " ".join(lines)
    txt = re.sub(r"\s", " ", txt)
    return txt.strip()

def _text_quality(txt: str) -> dict:
    n_chars = len(txt)
    n_alpha = sum(c.isalpha() for c in txt)
    n_words = len(txt.split())
    alpha_ratio = (n_alpha / max(n_chars, 1))
    return {
        "chars": n_chars,
        "words": n_words,
        "alpha_ratio": alpha_ratio,
        "looks_ok": (n_words >= 30 and alpha_ratio >= 0.10),
    }

def _ts():
    return dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")

def _sleep(sec: float, verbose: int):
    if sec > 0 and verbose:
        print(f"[{_ts()}] sleep {sec:.2f}s", file=sys.stderr)
    time.sleep(sec)

def _request(session, method, url, *, params=None, timeout=30, retries=5, backoff=1.6, verbose=0):
    attempt, delay = 0, 0.0
    while True:
        try:
            if verbose >= 2:
                print(f"[{_ts()}] {method.upper()} {url} params={params}", file=sys.stderr)
            r = session.request(method, url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            attempt += 1
            if attempt > retries:
                if verbose:
                    print(f"[warn] giving up after {retries} retries: {e}", file=sys.stderr)
                raise
            delay = backoff if delay == 0 else delay * backoff
            if verbose:
                print(f"[warn] {type(e).__name__}: {e}; retry {attempt}/{retries} in {delay:.2f}s", file=sys.stderr)
            _sleep(delay, verbose)

def _chunks(start: dt.date, end: dt.date, days: int) -> List[Tuple[dt.date, dt.date]]:
    out, cur = [], start
    while cur <= end:
        nxt = min(cur + dt.timedelta(days=days-1), end)
        out.append((cur, nxt)); cur = nxt + dt.timedelta(days=1)
    return out

def _scrape_ids(session, start: dt.date, end: dt.date, collections: List[str], *, retries, backoff, pause, timeout, verbose, max_items=None) -> List[Dict]:
    coll_ors = " OR ".join(f"collection:{c}" for c in collections)
    q = f"({coll_ors}) AND date:[{start} TO {end}]"
    params = {"q": q, "fields": "identifier,date,title", "sorts": "date asc,identifier asc", "count": 1000, "size": 1000}
    items, cursor = [], None
    while True:
        p = dict(params)
        if cursor: p["cursor"] = cursor
        last_err = None
        for base in SCRAPE_HOSTS:
            try:
                if verbose:
                    print(f"[{_ts()}] scrape host={base}", file=sys.stderr)
                r = _request(session, "get", base, params=p, timeout=timeout, retries=retries, backoff=backoff, verbose=verbose)
                obj = r.json()
                batch = obj.get("items", []) or []
                items.extend(batch)
                cursor = obj.get("cursor")
                if verbose:
                    print(f"[scrape] +{len(batch)} (total {len(items)}) cursor={'yes' if cursor else 'no'}", file=sys.stderr)
                break
            except Exception as e:
                last_err = e
                if verbose:
                    print(f"[warn] scrape failed at {base}: {e}", file=sys.stderr)
        if last_err and not items and not cursor:
            raise last_err
        if max_items and len(items) >= max_items:
            return items[:max_items]
        if not cursor: break
        _sleep(pause, verbose)
    return items

def _pick_caption_file(files_meta: List[Dict]) -> Optional[str]:
    names = [f.get("name","") for f in files_meta if f.get("name")]
    prefs = [
        r"(?i)\bcc5\.srt(\.gz)?$", r"(?i)\bcc5\.txt(\.gz)?$", r"(?i)\balign\.srt$",
        r"(?i)\.srt(\.gz)?$", r"(?i)\.vtt(\.gz)?$",
    ]
    for pat in prefs:
        for n in names:
            if re.search(pat, n):
                return n
    return None

def _download_caption(session, identifier: str, fname: str, *, retries, backoff, timeout, verbose) -> str:
    url = DL_URL.format(identifier=identifier, fname=quote(fname))
    r = _request(session, "get", url, timeout=timeout, retries=retries, backoff=backoff, verbose=verbose)
    data = r.content
    if fname.lower().endswith(".gz"):
        try: data = gzip.decompress(data)
        except OSError: pass
    try: return data.decode("utf-8", errors="ignore")
    except UnicodeDecodeError: return data.decode("latin-1", errors="ignore")

def _extract_transcript_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    full = soup.get_text(" ", strip=False)
    m = re.search(r"\b\d{1,2}:\d{2}\s?(am|pm)\b", full, re.I)  # time-of-day anchor
    start_ix = m.start() if m else 0
    stop_ix = len(full)
    for marker in (r"\bIN COLLECTIONS\b", r"\bBorrow Program\b", r"\bSIMILAR ITEMS\b"):
        m2 = re.search(marker, full)
        if m2: stop_ix = min(stop_ix, m2.start())
    chunk = full[start_ix:stop_ix]
    return re.sub(r"[ \t]+", " ", chunk)

def _fetch_transcript(session, identifier: str, *, retries, backoff, timeout, verbose) -> Tuple[str, Dict]:
    meta_url = META_URL.format(identifier=identifier)
    try:
        r = _request(session, "get", meta_url, timeout=timeout, retries=retries, backoff=backoff, verbose=verbose)
        meta = r.json()
        files = meta.get("files", []) or []
        cap = _pick_caption_file(files)
        if cap:
            if verbose:
                print(f"[meta] {identifier} caption file={cap}", file=sys.stderr)
            raw = _download_caption(session, identifier, cap, retries=retries, backoff=backoff, timeout=timeout, verbose=verbose)
            ext = cap.lower()
            norm = _normalize_caption_text(raw, ext_hint=ext) if (ext.endswith(".srt") or ext.endswith(".vtt") or ".srt" in ext or ".vtt" in ext) else raw
            q = _text_quality(norm)
            if verbose:
                print(f"[cap] {identifier} quality chars={q['chars']} words={q['words']} alpha={q['alpha_ratio']:.3f} ok={q['looks_ok']}", file=sys.stderr)
            if q["looks_ok"]:
                return norm, {"source":"caption_file", "caption_file":cap, "quality":q}
            else:
                if verbose:
                    print(f"[cap] {identifier} low-signal caption; fallback to HTML", file=sys.stderr)
        else:
            if verbose:
                print(f"[meta] {identifier} no downloadable captions; fallback to HTML", file=sys.stderr)
    except Exception as e:
        if verbose:
            print(f"[meta] {identifier} metadata failed: {e}; fallback to HTML", file=sys.stderr)

    det = DETAILS_URL.format(identifier=identifier)
    r2 = _request(session, "get", det, timeout=timeout, retries=retries, backoff=backoff, verbose=verbose)
    html_txt = _extract_transcript_from_html(r2.text)
    q2 = _text_quality(html_txt)
    if verbose:
        print(f"[html] {identifier} quality chars={q2['chars']} words={q2['words']} alpha={q2['alpha_ratio']:.3f} ok={q2['looks_ok']}", file=sys.stderr)
    return html_txt, {"source":"html_page", "quality":q2}

def _atomic_write_json_gz(path: str, obj: Dict):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)

def _dedupe_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _expand_channels_to_collections(codes: Tuple[str, ...]) -> List[str]:
    cols = []
    for code in codes:
        c = code.strip().upper()
        if not c:
            continue
        if c.startswith("TV-"):
            cols.append(c)
            continue
        if c.endswith("W"):
            cols.append(f"TV-{c}")
        else:
            cols.extend([f"TV-{c}", f"TV-{c}W"])
    return cols

@click.command()
@click.option("--start", type=click.DateTime(formats=["%Y-%m-%d"]), required=True)
@click.option("--end",   type=click.DateTime(formats=["%Y-%m-%d"]), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False), default="data_transcripts", show_default=True)
@click.option("--chunk-days", type=int, default=3, show_default=True)
@click.option("--pause", type=float, default=1.0, show_default=True)
@click.option("--retries", type=int, default=5, show_default=True)
@click.option("--backoff", type=float, default=1.6, show_default=True)
@click.option("--timeout", type=int, default=30, show_default=True)
@click.option("--max-items", type=int, default=None, help="limit per chunk for smoke tests")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True)
@click.option("-c", "--collections", multiple=True, help="IA collections (repeatable), e.g., TV-FOXNEWS, TV-CNNW, TV-MSNBCW.")
@click.option("-n", "--channels", multiple=True, help="Network codes (repeatable): e.g., FOXNEWS, FOXNEWSW, CNN, CNNW, MSNBC, MSNBCW. Prefixed to TV-; non-W codes also include the W variant.")
@click.option("-v", "--verbose", count=True)
def main(start, end, out_dir, chunk_days, pause, retries, backoff, timeout, max_items, overwrite, collections, channels, verbose):
    start_d, end_d = start.date(), end.date()

    # Build target collections
    cols = list(collections) if collections else []
    cols.extend(_expand_channels_to_collections(channels))
    if not cols:
        cols = ["TV-FOXNEWS", "TV-FOXNEWSW"]  # default

    cols = _dedupe_preserve_order(cols)
    cols_str = ", ".join(cols)
    print(f"[info] caching IA transcripts {start_d}..{end_d} from collections: {cols_str} -> {out_dir}", file=sys.stderr)

    session = requests.Session(); session.headers.update(UA)

    for ci, (d0, d1) in enumerate(_chunks(start_d, end_d, chunk_days), 1):
        print(f"[chunk {ci}] {d0}..{d1}", file=sys.stderr)
        items = _scrape_ids(session, d0, d1, cols, retries=retries, backoff=backoff, pause=pause, timeout=timeout, verbose=verbose, max_items=max_items)
        print(f"[chunk] {len(items)} items", file=sys.stderr)

        for i, it in enumerate(items, 1):
            ident = it.get("identifier","")
            day   = (it.get("date","") or "")[:10]
            title = it.get("title","")
            chan  = ident.split("_", 1)[0] if "_" in ident else ""
            out_path = os.path.join(out_dir, day, f"{ident}.json.gz")
            if not overwrite and os.path.exists(out_path):
                if verbose >= 2: print(f"[skip] exists {ident}", file=sys.stderr)
                continue

            if verbose:
                print(f"[{_ts()}] [{i}/{len(items)}] {ident} — transcript", file=sys.stderr)
            try:
                txt, meta = _fetch_transcript(session, ident, retries=retries, backoff=backoff, timeout=timeout, verbose=verbose)
                norm_wc = len(txt.split())
                rec = {
                    "identifier": ident,
                    "date": day,
                    "title": title,
                    "channel": chan,
                    "word_count": norm_wc,
                    "transcript": txt,
                    "source": meta.get("source"),
                    "caption_file": meta.get("caption_file"),
                    "quality": meta.get("quality"),
                    "details_url": DETAILS_URL.format(identifier=ident),
                    "sha1": hashlib.sha1(txt.encode("utf-8", errors="ignore")).hexdigest(),
                }
                _atomic_write_json_gz(out_path, rec)
                if verbose:
                    print(f"[write] {out_path} ({norm_wc} words)", file=sys.stderr)
            except Exception as e:
                print(f"[err] {ident}: {e}", file=sys.stderr)
            _sleep(pause, verbose)

    print("[done] fetch complete", file=sys.stderr)

if __name__ == "__main__":
    main()
