"""Command-line entry point: ``instant-notes <command>``.

Subcommands:
  start    run the background daemon (global hotkeys)
  notes    list / search stored notes
  bench    benchmark STT backends on a sample
  version  print the version
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from instant_notes import __version__
from instant_notes.config import load_config
from instant_notes.storage import Note, NoteStore
from instant_notes.stt.registry import available_backends, build_backend


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="instant-notes", description="Hotkey-driven voice notes for macOS."
    )
    parser.add_argument(
        "--config", default=None, help="path to a config.toml (optional)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("start", help="run the background daemon")

    p_notes = sub.add_parser("notes", help="list or search stored notes")
    p_notes.add_argument("--search", metavar="Q", default=None, help="FTS query")
    p_notes.add_argument("--today", action="store_true", help="only today's notes")
    p_notes.add_argument(
        "--limit", type=int, default=20, help="max notes to show (default 20)"
    )

    p_bench = sub.add_parser("bench", help="benchmark STT backends on a sample")
    p_bench.add_argument("--audio", required=True, help="path to a WAV sample")
    p_bench.add_argument(
        "--reference", default=None, help="path to a reference transcript (.txt)"
    )
    p_bench.add_argument(
        "--backends",
        default=None,
        help="comma-separated backend names (default: all available)",
    )

    sub.add_parser("version", help="print the version")
    return parser


def _cmd_start(args) -> int:
    from instant_notes.daemon import InstantNotesDaemon

    cfg = load_config(args.config)
    InstantNotesDaemon(cfg).run()
    return 0


def _format_note_line(note: Note) -> tuple[str, str, str]:
    ts = note.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
    return ts, note.source_backend or "?", note.text


def _cmd_notes(args) -> int:
    from rich.console import Console
    from rich.table import Table

    cfg = load_config(args.config)
    store = NoteStore(cfg.db_path)
    try:
        if args.search:
            notes = store.search(args.search, limit=args.limit)
        elif args.today:
            notes = store.today()[: args.limit]
        else:
            notes = store.recent(limit=args.limit)
    finally:
        store.close()

    console = Console()
    if not notes:
        console.print("[yellow]No notes found.[/yellow]")
        return 0

    table = Table(show_header=True, header_style="bold")
    table.add_column("When", style="cyan", no_wrap=True)
    table.add_column("Backend", style="green", no_wrap=True)
    table.add_column("Note", overflow="fold")
    for note in notes:
        ts, backend, text = _format_note_line(note)
        table.add_row(ts, backend, text)
    console.print(table)
    return 0


def _cmd_bench(args) -> int:
    from instant_notes.audio import load_wav
    from instant_notes.bench.harness import format_report, run_benchmark

    cfg = load_config(args.config)

    if not Path(args.audio).exists():
        print(f"audio file not found: {args.audio}")
        return 1
    audio = load_wav(args.audio)

    if args.backends:
        names = [n.strip() for n in args.backends.split(",") if n.strip()]
    else:
        names = available_backends(cfg)
    if not names:
        print("No backends available — set API keys or install faster-whisper.")
        return 1

    backends = []
    for name in names:
        try:
            backends.append(build_backend(name, cfg))
        except Exception as exc:
            print(f"[skip] {name}: {exc}")
    if not backends:
        print("No backends could be built.")
        return 1

    reference: str | None = None
    if args.reference:
        ref_path = Path(args.reference)
        if not ref_path.exists():
            print(f"reference file not found: {args.reference}")
            return 1
        reference = ref_path.read_text(encoding="utf-8").strip()

    results = run_benchmark(audio, backends, reference=reference)
    report = format_report(results)
    print(report)

    out_dir = Path("docs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{stamp}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {out_path}")
    return 0


def _cmd_version(args) -> int:
    print(__version__)
    return 0


_DISPATCH = {
    "start": _cmd_start,
    "notes": _cmd_notes,
    "bench": _cmd_bench,
    "version": _cmd_version,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _DISPATCH[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # handled error → exit 1
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
