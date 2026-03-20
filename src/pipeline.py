"""
Main pipeline orchestrator.
Flow: [scrape (optional)] → clean → nlp → visualize → strategy (Gemini) → email

Usage:
    python src/pipeline.py                                  # full pipeline, local data
    python src/pipeline.py --data-source apify              # fetch fresh data from Apify
    python src/pipeline.py --data-source apify --trigger    # trigger new Apify run
    python src/pipeline.py --skip email                     # skip email step
    python src/pipeline.py --only strategy                  # run single step
    python src/pipeline.py --start-from strategy            # resume from step
    python src/pipeline.py --dry-run                        # print commands only
"""
import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

STEPS_LOCAL = [
    ("clean",     "src/clean_data.py",    []),
    ("nlp",       "src/nlp_analysis.py",  []),
    ("visualize", "src/visualize.py",     []),
    ("strategy",  "src/strategy.py",      []),
    ("email",     "src/email_sender.py",  []),
]

STEPS_APIFY = [
    ("scrape",    "src/scraper.py",       ["--output", "data"]),
    ("clean",     "src/clean_data.py",    []),
    ("nlp",       "src/nlp_analysis.py",  []),
    ("visualize", "src/visualize.py",     []),
    ("strategy",  "src/strategy.py",      []),
    ("email",     "src/email_sender.py",  []),
]


def run_step(name: str, script: str, args: list, dry_run: bool = False) -> bool:
    cmd = [sys.executable, script] + args
    logger.info(f"\n{'=' * 60}")
    logger.info(f"STEP: {name}")
    logger.info(f"CMD:  {' '.join(cmd)}")
    logger.info(f"{'=' * 60}")

    if dry_run:
        logger.info("[DRY RUN] Skipping execution")
        return True

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error(f"Step '{name}' failed (exit code {result.returncode})")
        return False

    logger.info(f"Step '{name}' completed ✅")
    return True


def main():
    all_step_names = list({s[0] for s in STEPS_APIFY})

    parser = argparse.ArgumentParser(description="Hỏa Lò Facebook Analysis Pipeline")
    parser.add_argument(
        "--data-source",
        choices=["local", "apify"],
        default="local",
        help="'local' = use existing CSV files in data/; 'apify' = fetch from Apify"
    )
    parser.add_argument("--trigger", action="store_true", help="Trigger new Apify Actor run (with --data-source apify)")
    parser.add_argument("--start-from", choices=all_step_names, default=None)
    parser.add_argument("--stop-after", choices=all_step_names, default=None)
    parser.add_argument("--skip", nargs="*", default=[], help="Steps to skip")
    parser.add_argument("--only", choices=all_step_names, default=None, help="Run single step")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    steps = STEPS_APIFY if args.data_source == "apify" else STEPS_LOCAL

    if args.data_source == "apify" and args.trigger:
        for i, (name, script, step_args) in enumerate(steps):
            if name == "scrape":
                steps[i] = (name, script, step_args + ["--trigger"])

    logger.info(f"Data source: {args.data_source}")
    logger.info(f"Steps: {' → '.join(s[0] for s in steps)}")

    started = args.start_from is None
    for name, script, step_args in steps:
        if args.only and name != args.only:
            continue
        if not started:
            if name == args.start_from:
                started = True
            else:
                logger.info(f"Skipping: {name} (before --start-from)")
                continue
        if name in (args.skip or []):
            logger.info(f"Skipping: {name} (--skip)")
            continue

        success = run_step(name, script, step_args, args.dry_run)
        if not success:
            logger.error(f"Pipeline failed at step: {name}")
            sys.exit(1)

        if args.stop_after and name == args.stop_after:
            logger.info(f"Stopping after: {name}")
            break

    logger.info("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
