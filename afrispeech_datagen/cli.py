"""Command-line interface for the VoxCPM synthetic-speech generator.

Generate synthetic TTS training data from a text dataset, locally. Examples:

    # Preview 5 clips before a big run (needs a GPU)
    afrispeech-datagen --dataset ghananlpcommunity/some-text --text-column text --preview 5

    # Generate 5 hours, 50/50 male/female, into data/<name>
    afrispeech-datagen --dataset ghananlpcommunity/some-text --text-column text \\
        --hours 5 --name twi-run

    # Resume: just re-run the same command (skips finished rows)
    # Push the result to an HF dataset repo when done
    afrispeech-datagen --dataset … --text-column text --hours 5 --name twi-run \\
        --push you/my-synth
"""

from __future__ import annotations

import argparse
import os
import sys

from .generator import MODEL_ID, sanitize_name

DATASET_ORG = "AfriSpeech"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="afrispeech-datagen", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_argument_group("source")
    src.add_argument("--dataset", help="source text dataset id on the HF Hub")
    src.add_argument("--config", help="dataset config (optional)")
    src.add_argument("--split", default="train")
    src.add_argument("--text-column", help="column holding the text to synthesise")
    src.add_argument("--max-chars", type=int, default=400, help="skip rows longer than this")

    gen = p.add_argument_group("generation")
    gen.add_argument("--hours", type=float, default=1.0, help="target hours of audio")
    gen.add_argument("--voices", choices=["custom", "male", "female"], default="custom")
    gen.add_argument("--male-pct", type=int, default=50, help="%% male in custom mode")
    gen.add_argument("--instances", type=int, help="parallel model instances (default: auto by VRAM)")
    gen.add_argument("--cfg", type=float, default=2.0, dest="cfg_value", help="CFG value")
    gen.add_argument("--steps", type=int, default=10, help="inference timesteps")
    gen.add_argument("--model", default=MODEL_ID, help="VoxCPM model id")

    out = p.add_argument_group("output")
    out.add_argument("--out", help="output directory (default: data/<name>)")
    out.add_argument("--name", help="run name (folder under data/; enables resume)")
    out.add_argument("--save-every", type=int, default=200, help="write manifest every N rows")
    out.add_argument("--push", metavar="REPO_ID", help="upload the result to this HF dataset repo")
    out.add_argument("--private", action="store_true", help="make the pushed repo private")
    out.add_argument("--token", help="HF token (else HF_TOKEN env)")

    misc = p.add_argument_group("misc")
    misc.add_argument("--preview", type=int, metavar="N",
                      help="generate N preview clips and exit (no full run)")
    misc.add_argument("--list-datasets", action="store_true",
                      help=f"list datasets under the {DATASET_ORG} org and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if args.list_datasets:
        from huggingface_hub import HfApi
        ids = sorted(d.id for d in HfApi(token=args.token).list_datasets(author=DATASET_ORG, limit=500))
        print("\n".join(ids) if ids else f"(no datasets found under {DATASET_ORG})")
        return 0

    if not args.dataset or not args.text_column:
        sys.exit("Both --dataset and --text-column are required (see --help).")

    from . import generator

    # ---- preview ---------------------------------------------------------- #
    if args.preview:
        print(f"Generating {args.preview} preview clip(s)…", file=sys.stderr)
        name = args.name or sanitize_name(args.dataset.split("/")[-1])
        out_dir = args.out or os.path.join("data", name)
        clips = generator.preview(
            dataset=args.dataset, text_column=args.text_column, out_dir=out_dir,
            config=args.config, split=args.split, voices=args.voices, male_pct=args.male_pct,
            cfg_value=args.cfg_value, steps=args.steps, n=args.preview,
            max_chars=args.max_chars, model_id=args.model, token=args.token,
        )
        for c in clips:
            print(f"  [{c['gender']}] {c['duration']}s  {c['file']}\n      {c['text'][:90]}")
        return 0

    # ---- full run --------------------------------------------------------- #
    name = args.name or sanitize_name(args.dataset.split("/")[-1])
    out_dir = args.out or os.path.join("data", name)

    from tqdm.auto import tqdm
    bar = tqdm(total=round(args.hours * 3600), unit="s", unit_scale=False,
               desc="Synthesising audio", file=sys.stderr)
    state = {"last": 0.0}

    def _on_clip(total_sec):
        delta = total_sec - state["last"]
        if delta > 0:
            bar.update(delta)
            state["last"] = total_sec

    summary = generator.generate(
        dataset=args.dataset, text_column=args.text_column, out_dir=out_dir,
        config=args.config, split=args.split, target_hours=args.hours,
        voices=args.voices, male_pct=args.male_pct, instances=args.instances,
        cfg_value=args.cfg_value, steps=args.steps, max_chars=args.max_chars,
        model_id=args.model, token=args.token, save_every=args.save_every,
        on_clip=_on_clip, progress=lambda m: bar.set_description(m[:48]),
    )
    bar.close()

    print(f"\n✅ {summary['rows']} clips · {summary['hours']:.2f} h "
          f"({summary['errors']} errors) → {summary['out_dir']}", file=sys.stderr)
    print(f"   audio/  manifest.jsonl  progress.json", file=sys.stderr)

    if args.push:
        from huggingface_hub import HfApi, create_repo
        create_repo(args.push, repo_type="dataset", token=args.token,
                    private=args.private, exist_ok=True)
        HfApi(token=args.token).upload_folder(
            folder_path=out_dir, path_in_repo=os.path.basename(out_dir.rstrip("/")),
            repo_id=args.push, repo_type="dataset",
            commit_message=f"synthetic data: {summary['rows']} clips, {summary['hours']:.2f}h",
        )
        print(f"   pushed → https://huggingface.co/datasets/{args.push}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
