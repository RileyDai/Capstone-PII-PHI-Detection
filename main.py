import argparse
import re
from typing import Dict, List, Iterable, Tuple, Optional
import yaml
import sys, os
import json

# =========================
# Regex patterns 
# =========================
REGEX_PATTERNS: Dict[str, str] = {
    "Email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "Phone": r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}",
    "Ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "Address": r"\b\d{1,5}\s+[A-Za-z0-9.\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Highway|Hwy)\b\.?",
    "DriverLicense": r"\b([A-Z]{1,3}-?\d{3,9}|[A-Z]\d{6,8}|[A-Z]{2}\d{6})\b",
    "CreditCard": r"\b(?:\d[ -]*?){13,16}\b",
    "IP": r"\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(?:\.(?!$)|$)){4}\b"
          r"|\b(?:[A-F0-9]{1,4}:){7}[A-F0-9]{1,4}\b",
    "Date": r"\b(?:(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])"  # YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD
         r"|(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)\d{2}"      # DD-MM-YYYY / DD/MM/YYYY
         r"|(?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])[-/.](?:19|20)\d{2}"      # MM-DD-YYYY / MM/DD/YYYY
         r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s*(?:19|20)\d{2}"  # e.g. Jan 5, 2025
         r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s*(?:19|20)\d{2})\b",
    "Numbers": r"\b\d{1,6}(?:-\d{1,6}+\b)",
}


# Preset name for "select all"
REGEX_ALL_ALIAS = "ALL"

# =========================
# Built-in presets (levels)
# =========================
REGEX_PRESETS: Dict[str, List[str]] = {
    "none": [],
    "low": ["Email", "Phone"],
    "medium": ["Email", "Phone", "Ssn"],
    "high": ["Email", "Phone", "Ssn", "Address"],
    "strict": ["Email", "Phone", "Ssn", "Address", "DriverLicense", "CreditCard", "IP", "Date","Numbers"],  
}

SPACY_PRESETS: Dict[str, List[str]] = {
    "none": [],
    "low": ["PERSON","DATE"],
    "medium": ["PERSON", "ORG", "GPE","DATE","TIME"],
    "high": ["PERSON", "ORG", "GPE", "LOC","DATE","TIME"],
    "strict": [
        "PERSON","ORG","GPE","LOC","NORP","FAC","PRODUCT","EVENT",
        "WORK_OF_ART","LAW","LANGUAGE","DATE","TIME","MONEY","PERCENT",
        "CARDINAL","QUANTITY","ORDINAL"
    ],
}

def compile_selected_patterns(all_patterns: Dict[str, str],
                              selected_labels: Iterable[str]) -> List[Tuple[str, re.Pattern]]:
    """
    Build compiled regex list from selected labels.
    Supports 'ALL'; unknown labels are ignored.
    """
    selected = [x.strip() for x in selected_labels if x and x.strip()]
    if not selected:
        return []
    if any(lbl.upper() == REGEX_ALL_ALIAS for lbl in selected):
        return [(k, re.compile(v, flags=re.IGNORECASE)) for k, v in all_patterns.items()]
    out = []
    for lbl in selected:
        if lbl in all_patterns:
            out.append((lbl, re.compile(all_patterns[lbl], flags=re.IGNORECASE)))
    return out

def apply_regex_mask(text: str,
                     compiled_patterns: List[Tuple[str, re.Pattern]],
                     token: str) -> str:
    masked = text
    for _, pat in compiled_patterns:
        masked = pat.sub(token, masked)
    return masked

def apply_spacy_mask(text: str,
                     spacy_labels: Iterable[str],
                     token: str,
                     nlp) -> str:
    """
    Replace spaCy entities whose labels are in spacy_labels with token.
    We usually run regex first, then spaCy.
    """
    doc = nlp(text)
    spans = []
    wanted = set(lbl.strip() for lbl in spacy_labels if lbl and lbl.strip())
    for ent in doc.ents:
        if ent.label_ in wanted:
            spans.append((ent.start_char, ent.end_char))

    new_text = text
    for start, end in sorted(spans, key=lambda x: x[0], reverse=True):
        new_text = new_text[:start] + token + new_text[end:]
    return new_text

def load_spacy_model(model_name: str = "en_core_web_lg"):
    """
    Load spaCy model; if missing, print a hint and return None.
    """
    try:
        import spacy
        nlp = spacy.load(model_name)
        return nlp
    except Exception as e:
        print(f"[WARN] Failed to load spaCy model: {e}")
        print("       To enable NER, install it first:")
        print("       pip install spacy && python -m spacy download en_core_web_trf")
        return None

def resolve_labels(preset_name: str,
                   custom_labels: Iterable[str],
                   preset_table: Dict[str, List[str]],
                   all_keyword: Optional[str] = None,
                   all_pool: Optional[Iterable[str]] = None) -> List[str]:
    """
    Merge preset labels and custom labels (union).
    Supports an ALL keyword if provided.
    """
    result = set()
    # Preset
    preset_name = (preset_name or "none").lower()
    if preset_name in preset_table:
        result.update(preset_table[preset_name])

    # Custom
    for lbl in (custom_labels or []):
        lbl = lbl.strip()
        if not lbl:
            continue
        if all_keyword and lbl.upper() == all_keyword:
            if all_pool:
                result.update(all_pool)
        else:
            result.add(lbl)

    return sorted(result)

def mask_pii(text: str,
             regex_labels: Iterable[str],
             spacy_labels: Iterable[str],
             replace_token: str = "[SENSITIVE]",
             disable_regex: bool = False,
             disable_spacy: bool = False,
             nlp=None) -> str:
    """
    Main redaction pipeline: Regex (optional) -> spaCy (optional)
    """
    out = text

    if not disable_regex:
        pats = compile_selected_patterns(REGEX_PATTERNS, regex_labels)
        out = apply_regex_mask(out, pats, replace_token)

    if not disable_spacy and spacy_labels:
        if nlp is None:
            nlp = load_spacy_model()
        if nlp is not None:
            out = apply_spacy_mask(out, spacy_labels, replace_token, nlp)
        else:
            print("[INFO] Skipped spaCy masking.")
    return out

def read_text_from_sources(input_text: Optional[str],
                           input_file: Optional[str]) -> str:
    if input_text:
        return input_text
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            return f.read()

    sample_text = """
John Smith lives in New York City, at 123 Main Ave.
He works for Google, and his email is john.smith@example.com.
Call him at +1 (415) 555-9876.
"""
    return sample_text

def write_output_if_needed(text: str, output_file: Optional[str]) -> None:
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[OK] Written to file: {output_file}")

def list_spacy_labels(nlp) -> None:
    try:
        labels = nlp.get_pipe("ner").labels
        print("spaCy NER labels:", ", ".join(labels))
    except Exception:
        print("No NER component found in the current spaCy pipeline.")

def _load_config_from_cli() -> dict:
    """
    If '--config path' is present, load YAML/JSON and remove the flag
    from sys.argv so argparse won't see it.
    """
    if "--config" in sys.argv:
        i = sys.argv.index("--config")
        if i + 1 >= len(sys.argv):
            raise SystemExit("[ERROR] --config requires a file path")
        path = sys.argv[i + 1]
        # remove the two args so argparse doesn't error out
        del sys.argv[i:i + 2]

        if not os.path.exists(path):
            raise SystemExit(f"[ERROR] Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            if path.endswith((".yaml", ".yml")):
                return yaml.safe_load(f) or {}
            elif path.endswith(".json"):
                return json.load(f) or {}
            else:
                raise SystemExit("[ERROR] Config must be .yaml/.yml or .json")
    return {}


def parse_args():
    _cfg = _load_config_from_cli()

    p = argparse.ArgumentParser(
        description="PII/PHI text redaction (regex + spaCy; multiple I/O channels)"
    )
    # Input
    p.add_argument("--input-text", type=str, default=None,
                   help="Process a literal text string (mutually exclusive with --input-file).")
    p.add_argument("--input-file", type=str, default=None,
                   help="Read text from a .txt file.")
    # Output
    p.add_argument("--output-file", type=str, default=None,
                   help="Write processed text to a .txt file (optional).")
    # Presets + custom labels
    p.add_argument("--regex-preset", type=str, default="high",
                   help="Regex preset level: none|low|medium|high|strict.")
    p.add_argument("--spacy-preset", type=str, default="low",
                   help="spaCy preset level: none|low|medium|high|strict.")
    p.add_argument("--regex-labels", type=str, default="",
                   help=f"Custom regex labels (comma-separated). Also supports '{REGEX_ALL_ALIAS}'.")
    p.add_argument("--spacy-labels", type=str, default="",
                   help="Custom spaCy labels (comma-separated).")
    # Toggles
    p.add_argument("--disable-regex", action="store_true",
                   help="Disable regex redaction.")
    p.add_argument("--disable-spacy", action="store_true",
                   help="Disable spaCy redaction.")
    # Others
    p.add_argument("--replace-token", type=str, default="[SENSITIVE]",
                   help="Replacement token.")
    p.add_argument("--spacy-model", type=str, default="en_core_web_sm",
                   help="spaCy model name.")
    p.add_argument("--list-spacy-labels", action="store_true",
                   help="List spaCy NER labels and exit.")
    
    args = p.parse_args()

    # NEW: overlay values from config onto parsed args (CLI still has higher priority)
    for k, v in _cfg.items():
        if hasattr(args, k) and v is not None:
            setattr(args, k, v)

    return args


def main():
    args = parse_args()

    # Load spaCy model if needed
    nlp = None
    if not args.disable_spacy:
        nlp = load_spacy_model(args.spacy_model)
        if nlp and args.list_spacy_labels:
            list_spacy_labels(nlp)
            return
        if (not nlp) and args.list_spacy_labels:
            return

    # Read input
    raw_text = read_text_from_sources(args.input_text, args.input_file)

    # Resolve final label sets (preset ∪ custom)
    custom_regex = [s.strip() for s in args.regex_labels.split(",")] if args.regex_labels else []
    final_regex_labels = resolve_labels(
        preset_name=args.regex_preset,
        custom_labels=custom_regex,
        preset_table=REGEX_PRESETS,
        all_keyword=REGEX_ALL_ALIAS,
        all_pool=REGEX_PATTERNS.keys()
    )

    custom_spacy = [s.strip() for s in args.spacy_labels.split(",")] if args.spacy_labels else []
    final_spacy_labels = resolve_labels(
        preset_name=args.spacy_preset,
        custom_labels=custom_spacy,
        preset_table=SPACY_PRESETS
    )

    # Process
    masked = mask_pii(
        text=raw_text,
        regex_labels=final_regex_labels,
        spacy_labels=final_spacy_labels,
        replace_token=args.replace_token,
        disable_regex=args.disable_regex,
        disable_spacy=args.disable_spacy,
        nlp=nlp
    )

    # Print to console
    print("=== Original Text ===")
    print(raw_text)
    print("\n=== Processed Text ===")
    print(masked)

    # Optionally write file
    write_output_if_needed(masked, args.output_file)

if __name__ == "__main__":
    main()
