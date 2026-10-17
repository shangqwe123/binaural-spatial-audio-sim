"""Build source manifests (JSONL) for generate_dataset.py.

Every output line: {"audio": path, "type": "speech"|"sfx", "text": ..., "group": ...}
plus dataset-specific fields. "group" decides the train/validation/test split
(speaker for Emilia, YouTube ID for AudioCaps).

  python prepare_sources.py emilia --root EMILIA_DIR --output sources/speech.jsonl
  python prepare_sources.py audiocaps --csv train.csv val.csv test.csv \
      --audio-dir AUDIOCAPS_AUDIO --output sources/sfx.jsonl
"""
import argparse
import csv
import json
import re
from pathlib import Path

AUDIO_SUFFIXES = {'.wav', '.flac', '.mp3', '.ogg'}
# Rough single-source filter for captions; BEWO-1M filtered similarly before a
# CLAP check. Anything joined by these words likely has two or more events.
MULTI_SOURCE = re.compile(
    r'\b(and|while|as|then|followed by|before|after|with|along with|over|'
    r'in the background|together|alternat\w*|interspersed|accompanied)\b', re.I)


def emilia(args):
    """Emilia / Emilia-Large after untarring: per-utterance .json next to .mp3.

    JSON keys used: text, speaker, language, duration, dnsmos, (wav).
    """
    root = args.root
    rows, skipped = [], 0
    for meta_path in sorted(root.rglob('*.json')):
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            skipped += 1
            continue
        if not isinstance(meta, dict) or 'text' not in meta:
            skipped += 1
            continue
        audio = next((meta_path.with_suffix(s) for s in ('.mp3', '.wav', '.flac')
                      if meta_path.with_suffix(s).is_file()), None)
        if audio is None and meta.get('wav') and (root/meta['wav']).is_file():
            audio = root/meta['wav']
        duration = float(meta.get('duration', 0) or 0)
        dnsmos = float(meta.get('dnsmos', 0) or 0)
        if (audio is None or not args.min_duration <= duration <= args.max_duration
                or dnsmos < args.min_dnsmos):
            skipped += 1
            continue
        rows.append({'audio': str(audio.resolve()), 'type': 'speech', 'text': meta['text'].strip(),
                     'group': str(meta.get('speaker') or meta_path.parent.name),
                     'language': meta.get('language'), 'duration': duration, 'dnsmos': dnsmos,
                     'source_dataset': 'emilia', 'source_id': meta.get('id', meta_path.stem)})
    return rows, skipped


def audiocaps(args):
    """AudioCaps CSVs (audiocap_id, youtube_id, start_time, caption) + downloaded audio.

    Audio files are matched by stem: youtube_id, Y<youtube_id>, audiocap_id,
    or <youtube_id>_<start_time...>.
    """
    index = {}
    for f in args.audio_dir.rglob('*'):
        if f.suffix.lower() in AUDIO_SUFFIXES:
            stem = f.stem[1:] if f.stem.startswith('Y') and len(f.stem) >= 12 else f.stem
            index.setdefault(f.stem, f)
            index.setdefault(stem, f)
            # YouTube IDs are 11 characters and may contain '_': <id>_<start>.wav
            if len(stem) > 11 and stem[11] == '_':
                index.setdefault(stem[:11], f)
    rows, skipped, multi = [], 0, 0
    for csv_path in args.csv:
        with csv_path.open(encoding='utf-8') as fh:
            for item in csv.DictReader(fh):
                audio = next((index[k] for k in (item['youtube_id'], 'Y'+item['youtube_id'],
                                                 item['audiocap_id']) if k in index), None)
                caption = item['caption'].strip()
                single = not MULTI_SOURCE.search(caption)
                if audio is None:
                    skipped += 1
                    continue
                if not single and not args.keep_multi:
                    multi += 1
                    continue
                rows.append({'audio': str(audio.resolve()), 'type': 'sfx', 'text': caption,
                             'group': item['youtube_id'], 'single_source_guess': single,
                             'source_dataset': 'audiocaps', 'source_id': item['audiocap_id'],
                             'split_hint': csv_path.stem})
    print(f'AudioCaps: dropped {multi} captions that look multi-source')
    return rows, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='dataset', required=True)
    e = sub.add_parser('emilia')
    e.add_argument('--root', type=Path, required=True)
    e.add_argument('--min-duration', type=float, default=2.0)
    e.add_argument('--max-duration', type=float, default=20.0)
    e.add_argument('--min-dnsmos', type=float, default=3.2)
    a = sub.add_parser('audiocaps')
    a.add_argument('--csv', type=Path, nargs='+', required=True)
    a.add_argument('--audio-dir', type=Path, required=True)
    a.add_argument('--keep-multi', action='store_true', help='Keep captions that look multi-source')
    for s in (e, a):
        s.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows, skipped = (emilia if args.dataset == 'emilia' else audiocaps)(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(f'{args.dataset}: wrote {len(rows)} sources to {args.output}; skipped {skipped}')


if __name__ == '__main__':
    main()
