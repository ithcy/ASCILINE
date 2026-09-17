"""
asgrabber.py
============
Grabs a single frame from a video at a given timestamp and renders it as
colored ASCII art, saved to a file. Output format is inferred from the
file extension:

  .html / .htm         -> standalone HTML page (ANSI colors -> <span> tags)
  anything else         -> raw ANSI-colored text (view with `cat file`)

Timestamp formats accepted: "SS", "SS.mmm", "MM:SS", "MM:SS.mmm",
"HH:MM:SS", "HH:MM:SS.mmm"
"""
import argparse
import html
import random
import re
import shutil
import sys

from ascii_video_player2 import (
    EMOJI_PRESETS, PALETTE_PRESETS, AsciiMapper, VideoDecoder, palette_cell_width,
)

_ANSI_COLOR_RE = re.compile(r'\x1b\[38;2;(\d+);(\d+);(\d+)m')
_ANSI_RESET_RE = re.compile(r'\x1b\[0m')


def _positive_float(value: str) -> float:
    f = float(value)
    if f <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return f


def _non_negative_float(value: str) -> float:
    f = float(value)
    if f < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return f


def _non_negative_int(value: str) -> int:
    i = int(value)
    if i < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return i


def _positive_int(value: str) -> int:
    i = int(value)
    if i <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return i


def parse_timestamp(ts: str) -> float:
    """Parses "HH:MM:SS.mmm" / "MM:SS" / "SS.mmm" / etc into seconds."""
    parts = ts.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid timestamp: {ts!r}")
    parts = [float(p) for p in parts]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


MAX_SAFE_COLS = 160  # Windows terminal often struggles above 160 (single-width) cols


def resolve_cols(cols: int, palette, to_terminal: bool) -> int:
    """Resolves the requested column count.

    cols > 0            -> used as-is.
    cols <= 0, to a live terminal (no --output file) -> auto-fits the
        current $COLUMNS, halved for wide/emoji palettes.
    cols <= 0, writing to a file -> a fixed default of 200 (no terminal to
        fit, since there's nothing live being displayed).
    """
    if cols > 0:
        return cols
    cell_width = palette_cell_width(palette or AsciiMapper.DEFAULT_PALETTE)
    if to_terminal:
        term_cols = shutil.get_terminal_size(fallback=(220, 50)).columns
        return max(1, min(term_cols // cell_width, MAX_SAFE_COLS // cell_width))
    return 200


def compute_grid(vid_w: int, vid_h: int, cols: int, palette, char_ratio: float = 0.45) -> tuple[int, int]:
    """Aspect-ratio-preserving grid size, scaled for wide/emoji palette glyphs."""
    cell_width = palette_cell_width(palette or AsciiMapper.DEFAULT_PALETTE)
    aspect = vid_h / vid_w
    rows = max(1, int(cols * aspect * char_ratio * cell_width))
    return cols, rows


def grab_frame(video: str, timestamp: float, cols: int, palette, quantize_bits: int,
               brightness: float = 1.0, saturation: float = 1.0,
               contrast: float = 1.0, gamma: float = 1.0) -> str:
    """Seeks to `timestamp` seconds and returns a colored ANSI frame string."""
    decoder = VideoDecoder(video, 2, 2)
    try:
        grid_cols, grid_rows = compute_grid(decoder.vid_w, decoder.vid_h, cols, palette)
        decoder._size = (grid_cols, grid_rows)
        decoder.seek(timestamp)
        gray, bgr = next(decoder)
    except StopIteration:
        raise ValueError(f"No frame found at timestamp {timestamp}s (past end of video?)")
    finally:
        decoder.release()

    mapper = AsciiMapper(palette, quantize_bits, brightness, saturation, contrast, gamma)
    return mapper.convert(gray, bgr)


def _class_name(index: int) -> str:
    """Short a, b, ..., z, aa, ab, ... class name for the given index."""
    chars = "abcdefghijklmnopqrstuvwxyz"
    name = ""
    index += 1
    while index > 0:
        index, rem = divmod(index - 1, 26)
        name = chars[rem] + name
    return name


def ansi_to_html(ascii_frame: str, font_size: int = 20) -> str:
    """Converts a True-Color-ANSI ASCII frame into a standalone HTML page.

    Repeated colors are deduped into CSS classes (instead of a `style=`
    attribute per run) and runs use `<b>` instead of `<span>` — both cut
    output size, though quantizing color with `--quality 1/2` (fewer unique
    colors overall) shrinks it far more than either of these.
    """
    body = _ANSI_RESET_RE.sub("", ascii_frame)
    parts = _ANSI_COLOR_RE.split(body)

    class_of: dict[str, str] = {}
    css_rules = []
    chunks = [html.escape(parts[0])] if parts[0] else []
    i = 1
    while i < len(parts):
        r, g, b = parts[i], parts[i + 1], parts[i + 2]
        text = parts[i + 3] if i + 3 < len(parts) else ""
        hex_color = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        cls = class_of.get(hex_color)
        if cls is None:
            cls = _class_name(len(class_of))
            class_of[hex_color] = cls
            css_rules.append(f".{cls}{{color:{hex_color}}}")
        chunks.append(f'<b class="{cls}">{html.escape(text)}</b>')
        i += 4
    body_html = "".join(chunks)

    return (
        "<!DOCTYPE html>\n"
        '<html><head><meta charset="utf-8"><title>ASCII Frame</title>\n'
        "<style>\n"
        "  body { background:#000; margin:0; padding:1em; }\n"
        f"  pre {{ color:#fff; font-family:monospace; font-size:{font_size}px; "
        "font-weight:bold; line-height:1; white-space:pre; }\n"
        "  " + "".join(css_rules) + "\n"
        "</style></head><body><pre>" + body_html + "</pre></body></html>\n"
    )


def main():
    parser = argparse.ArgumentParser(
        prog="asgrabber",
        description="Grab a single colored ASCII frame from a video at a timestamp.",
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "timestamp",
        help='Timestamp: "SS", "SS.mmm", "MM:SS", "MM:SS.mmm", "HH:MM:SS", "HH:MM:SS.mmm"',
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output file path. Format inferred from extension: "
             ".html/.htm -> HTML page, anything else -> raw ANSI text. "
             "Omit to print raw ANSI to stdout.",
    )
    parser.add_argument("-w", "--width", dest="cols", type=_non_negative_int, default=0,
        help="Grid width in characters. 0 -> auto-fits terminal (if printing "
             "to stdout) or 200 (if writing to a file) (default: 0)")
    parser.add_argument("-q", "--quality", type=int, choices=[0, 1, 2, 3], default=0,
        help="Color quality: 0=max quality, 3=max speed (default: 0)")
    parser.add_argument("--palette", default=None,
        help="Custom character palette, space-separated")
    parser.add_argument("--block", action="store_true", default=False,
        help='Shorthand for --palette " ░▒▓█" (Unicode shade blocks)')
    parser.add_argument("--preset", choices=list(PALETTE_PRESETS), default=None,
        help="Use a named preset palette")
    parser.add_argument("-r", "--random", action="store_true", default=False,
        help=f"Pick a random preset palette ({', '.join(PALETTE_PRESETS)}); "
             "excludes emoji presets unless --include-emoji is given")
    parser.add_argument("--include-emoji", action="store_true", default=False,
        help="Allow -r/--random to pick emoji presets too")
    parser.add_argument("--font-size", type=_positive_int, default=20,
        help="HTML output only: font size in px (default: 20)")
    parser.add_argument("-b", "--brightness", type=_non_negative_float, default=1.07,
        help="Color brightness multiplier, e.g. 1.5 = 50%% brighter (default: 1.07)")
    parser.add_argument("-s", "--saturation", type=_non_negative_float, default=1.3,
        help="Color saturation, 0=grayscale, 1=unchanged, >1=more vivid (default: 1.3)")
    parser.add_argument("-c", "--contrast", type=_non_negative_float, default=1.5,
        help="Contrast multiplier around the midpoint (default: 1.5)")
    parser.add_argument("-g", "--gamma", type=_positive_float, default=1.4,
        help="Gamma correction (>0), >1=lifts shadows/brighter, <1=darker (default: 1.4)")

    args = parser.parse_args()

    try:
        seconds = parse_timestamp(args.timestamp)
    except ValueError as e:
        parser.error(str(e))

    if sum([args.block, bool(args.palette), bool(args.preset), args.random]) > 1:
        parser.error("--block, --palette, --preset, and --random/-r are mutually exclusive")
    if args.random:
        pool = list(PALETTE_PRESETS) if args.include_emoji else [
            n for n in PALETTE_PRESETS if n not in EMOJI_PRESETS
        ]
        preset_name = random.choice(pool)
        custom_palette = PALETTE_PRESETS[preset_name]
        print(f"[+] Random palette: {preset_name}", file=sys.stderr)
    elif args.preset:
        custom_palette = PALETTE_PRESETS[args.preset]
    elif args.block:
        custom_palette = list(" ░▒▓█")
    elif args.palette:
        custom_palette = args.palette.split()
    else:
        custom_palette = None

    resolved_cols = resolve_cols(args.cols, custom_palette, to_terminal=args.output is None)

    try:
        frame = grab_frame(args.video, seconds, resolved_cols, custom_palette, args.quality,
                           args.brightness, args.saturation, args.contrast, args.gamma)
    except (FileNotFoundError, ValueError) as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        print(frame)
        return

    ext = args.output.rsplit(".", 1)[-1].lower() if "." in args.output else ""
    if ext in ("html", "htm"):
        content = ansi_to_html(frame, font_size=args.font_size)
    else:
        content = frame

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[+] Saved frame at {args.timestamp} ({seconds:.3f}s) -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
