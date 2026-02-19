#!/usr/bin/env python3
"""
format_json_to_csv.py

Scan JSON files in a folder and create a CSV with columns:
 - "Employer Name"   <- input_name
 - "Industry"        <- classification['label']
 - "Linkedin Profile" <- linkedin_url (top-level)

Usage:
    python format_json_to_csv.py
    python format_json_to_csv.py --input-dir data/output --output-file data/formatted_employers.csv
"""

from pathlib import Path
import json
import csv
import argparse
import sys

def extract_label(classification):
    """Safely extract a label from classification which might be None, a dict, or a list."""
    if classification is None:
        return ""
    if isinstance(classification, dict):
        return classification.get("label", "") or ""
    if isinstance(classification, list) and classification:
        # take first element if it's a dict
        first = classification[0]
        if isinstance(first, dict):
            return first.get("label", "") or ""
    return ""

def process_file(path: Path):
    """
    Read a JSON file and extract:
      - input_name
      - linkedin_url (top-level)
      - classification['label']
    Returns a tuple (input_name, industry_label, linkedin_url)
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # fallback to latin1 if file is not utf-8
        text = path.read_text(encoding="latin1")

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON in {path}: {e}", file=sys.stderr)
        return None

    input_name = obj.get("input_name", "") or ""
    linkedin_url = obj.get("linkedin_url", "") or ""
    classification = obj.get("classification")
    industry = extract_label(classification)

    return input_name, industry, linkedin_url

def main(input_dir: str, output_file: str, pattern: str = "*.json"):
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    json_files = sorted(input_path.glob(pattern))
    if not json_files:
        print(f"No JSON files found in {input_dir} matching {pattern}", file=sys.stderr)
        return 1

    rows = []
    for p in json_files:
        result = process_file(p)
        if result is None:
            # Skip malformed file but continue
            continue
        rows.append(result)

    # Write CSV
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Employer Name", "Industry", "Linkedin Profile"])
        for input_name, industry, linkedin_url in rows:
            writer.writerow([input_name, industry, linkedin_url])

    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Format JSON output files to a single CSV.")
    ap.add_argument(
        "--input-dir",
        default="data/output",
        help="Directory containing JSON files (default: data/output)",
    )
    ap.add_argument(
        "--output-file",
        default="data/formatted_employers.csv",
        help="Output CSV path (default: data/formatted_employers.csv)",
    )
    ap.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for files (default: *.json)",
    )
    args = ap.parse_args()
    raise SystemExit(main(args.input_dir, args.output_file, args.pattern))
