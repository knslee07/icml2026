#!/usr/bin/env python3
"""
Multi-Stage Extraction Pipeline for Compensation Consultant NER

This script implements the four-stage extraction pipeline described in the paper:
    1. Document Processing: Keyword retrieval + chunking
    2. Extraction: LLM-based entity extraction
    3. Post-processing: Validation + name normalization
    4. Aggregation: Cross-chunk deduplication

Paper: "Domain-Specific Alignment for Information Extraction:
        A Framework for Social Science Data Collection"

USAGE:
    python inference.py --adapter <path_to_lora_adapter> --data <eval_folder> --output results.json

    # Vanilla (base model, no adapter)
    python inference.py --vanilla --data <eval_folder> --output results.json
"""

import os
import re
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Tuple
import html
import chardet

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# Configuration
# =============================================================================

# Retrieval settings
WINDOW_SIZE = 250          # Characters around keyword matches
MAX_CHUNK_TOKENS = 4000    # Tokens per LLM call
CHUNK_OVERLAP = 25         # Token overlap between chunks

# Keyword patterns for retrieval (hierarchical: specific -> general)
PATTERNS_SPECIFIC = [
    "compensation consult", "compensation advis",
    "independent consult", "independent advis",
    "committee consult", "committee advis",
    "outside consult", "outside advis",
    "external consult", "external advis",
    "remuneration consult", "remuneration advis",
]

PATTERNS_COMMITTEE = [
    "compensation committee", "benefits committee",
    "remuneration committee", "HR Committee",
    "human resources committee",
]

PATTERNS_GENERAL = [
    "executive compensation",
]

ALL_PATTERNS = [PATTERNS_SPECIFIC, PATTERNS_COMMITTEE, PATTERNS_GENERAL]

# Instruction prompts
SHORT_INSTRUCTION = """You are a meticulous reader of proxy-statement excerpts.

TASK: Identify two categories of compensation consultants:
  1. RET: Consultants retained/engaged to advise on executive compensation.
  2. SURV: Survey/data providers NOT explicitly retained as advisors.

RULES:
- RET and SURV are mutually exclusive
- If consultant replaced during the period, list ONLY the final consultant
- Exclude: directors-only consultants, law firms, internal HR staff, public data sources
- If name undisclosed: 'UNNAMED'. If none found: 'NO_CONSULTANTS'

OUTPUT FORMAT:
{RET: 'Firm A', 'Firm B'}, {SURV: 'Firm C'}
- Use longest name variant, alphabetical order, single quotes
"""

LONG_INSTRUCTION = """You are a meticulous reader of proxy-statement excerpts.

TASK: Identify two categories of compensation consultants:
  1. RET: Consultants retained/engaged by committee/management/Board to advise on executive compensation.
     - Includes: advisory, analysis, design, market studies, recommendations, reviews.
  2. SURV: Survey/data providers for executive compensation NOT explicitly retained as advisors.
     - Includes: proprietary surveys (Radford, Mercer, Towers Watson).
     - RET and SURV are mutually exclusive.

CRITICAL RULES:
- Consultant replacements: List ONLY FINAL consultant, NOT predecessor.
- Director compensation only: Exclude (NO_CONSULTANTS)
- Fee amounts: IRRELEVANT (RET if retained regardless of fee)
- Internal employees: Exclude HR staff, executives
- Law firms: Exclude legal advisors

EXCLUSIONS (NO_CONSULTANTS for both):
- Public websites: Salary.com, Glassdoor, Indeed, LinkedIn Salary
- Proxy advisors: ISS, Glass Lewis
- Research orgs: Conference Board, WorldAtWork, NACD

OUTPUT FORMAT:
{RET: 'Firm A', 'Firm B'}, {SURV: 'Firm C'}
- Use longest name variant, alphabetical order, single quotes
- If none found: 'NO_CONSULTANTS'. If name undisclosed: 'UNNAMED'

TIMING: Filing year OR prior year only. If consultant replaced, list only final consultant.
"""


# =============================================================================
# Document Processing (Stage 1)
# =============================================================================

def detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def read_file(file_path: str) -> str:
    """Read file with automatic encoding detection."""
    try:
        encoding = detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()


def clean_text(text: str) -> str:
    """Clean HTML entities and normalize whitespace."""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)       # Normalize whitespace
    return text.strip()


def extract_header_info(text: str) -> Dict:
    """Extract metadata from SEC filing header."""
    info = {}
    patterns = {
        'filing_date': r'FILED AS OF DATE:\s+(\d{8})',
        'company_name': r'COMPANY CONFORMED NAME:\s+(.+?)(?=\n)',
        'cik': r'CENTRAL INDEX KEY:\s+(\d+)'
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text[:1000])
        if match:
            info[key] = match.group(1).strip()
    if 'filing_date' in info:
        info['year'] = info['filing_date'][:4]
    return info


def find_relevant_excerpts(text: str, window: int = WINDOW_SIZE) -> List[Dict]:
    """
    Stage 1a: Keyword-based retrieval.

    Find text sections containing compensation consultant keywords.
    Uses hierarchical pattern matching (specific -> general).
    """
    cleaned = clean_text(text)
    excerpts = []

    for pattern_group in ALL_PATTERNS:
        for pattern in pattern_group:
            for match in re.finditer(pattern, cleaned, re.IGNORECASE):
                start = max(0, match.start() - window)
                end = min(len(cleaned), match.end() + window)
                excerpts.append({
                    'text': cleaned[start:end],
                    'position': start,
                    'pattern': pattern
                })

    return merge_overlapping(excerpts)


def merge_overlapping(excerpts: List[Dict]) -> List[Dict]:
    """Merge overlapping text excerpts."""
    if not excerpts:
        return []

    sorted_exc = sorted(excerpts, key=lambda x: x['position'])
    merged = [sorted_exc[0]]

    for exc in sorted_exc[1:]:
        last = merged[-1]
        last_end = last['position'] + len(last['text'])

        if exc['position'] <= last_end:
            # Merge overlapping
            if exc['position'] + len(exc['text']) > last_end:
                extension = exc['text'][last_end - exc['position']:]
                last['text'] += extension
        else:
            merged.append(exc)

    return merged


def create_chunks(text: str, tokenizer, max_tokens: int = MAX_CHUNK_TOKENS,
                  overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Stage 1b: Chunk text into LLM-sized pieces.

    Creates overlapping chunks to avoid cutting sentences.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    for i in range(0, len(ids), max_tokens - overlap):
        chunk_ids = ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

    return chunks


# =============================================================================
# Extraction (Stage 2)
# =============================================================================

def extract_from_chunk(text: str, model, tokenizer, filing_year: str = None,
                       instruction: str = "long", raw_mode: bool = False) -> Dict[str, List[str]]:
    """
    Stage 2: LLM-based extraction.

    Query the model to extract RET and SURV consultants from a text chunk.
    """
    if not text:
        return {'RET': [], 'SURV': []}

    # Select instruction
    instr = LONG_INSTRUCTION if instruction == "long" else SHORT_INSTRUCTION

    # Build prompt
    prompt_parts = [instr]
    if filing_year:
        prompt_parts.append(
            f"\nTEMPORAL CONTEXT: This document was filed in {filing_year}. "
            f"Consultants retained in {filing_year} or {int(filing_year)-1} are considered 'currently retained'.\n"
        )
    prompt_parts.append(f"\nTEXT TO ANALYZE:\n{text}")
    prompt_parts.append(
        "\n\nAfter reading, output ONLY one line:\n"
        "{RET: 'Firm A', 'Firm B'}, {SURV: 'Firm C'}\n"
        "BEGIN OUTPUT:\n"
    )
    prompt = "".join(prompt_parts)

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=5120).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Parse response
    result = parse_llm_response(response)

    # Validate (skip for raw/vanilla mode)
    if not raw_mode:
        result = validate_against_text(result, text)

    return result


def parse_llm_response(response: str) -> Dict[str, List[str]]:
    """Parse LLM output into RET and SURV lists."""
    # Try strict format: {RET: ...}, {SURV: ...}
    pattern = r"\{RET:\s*([^}]*)\},\s*\{SURV:\s*([^}]*)\}"
    match = re.search(pattern, response, re.I)

    if match:
        ret = extract_quoted_names(match.group(1))
        surv = extract_quoted_names(match.group(2))
        return {'RET': sorted(set(ret)), 'SURV': sorted(set(surv))}

    # Fallback: return empty
    return {'RET': [], 'SURV': []}


def extract_quoted_names(text: str) -> List[str]:
    """Extract single-quoted names from text."""
    names = []
    in_quote = False
    current = ""

    for char in text:
        if char == "'" and not in_quote:
            in_quote = True
            current = ""
        elif char == "'" and in_quote:
            in_quote = False
            if current.strip() and current.strip().upper() != "NO_CONSULTANTS":
                names.append(current.strip())
        elif in_quote:
            current += char

    return names


# =============================================================================
# Post-Processing (Stage 3)
# =============================================================================

def validate_against_text(result: Dict[str, List[str]], source_text: str) -> Dict[str, List[str]]:
    """
    Stage 3a: Validate extracted names against source text.

    Reject hallucinations - names that don't appear in the source.
    """
    validated = {'RET': [], 'SURV': []}
    source_lower = source_text.lower()

    for category in ['RET', 'SURV']:
        for name in result.get(category, []):
            if name.upper() in ['UNNAMED', 'NO_CONSULTANTS']:
                validated[category].append(name)
            elif find_name_in_text(name, source_lower):
                validated[category].append(name)

    return validated


def find_name_in_text(name: str, text_lower: str) -> bool:
    """Check if a consultant name appears in text."""
    # Tokenize name
    words = re.findall(r'[a-z0-9&.]+', name.lower())
    words = [w for w in words if len(w) > 1 or w in ['&', '.']]

    if not words:
        return False

    # Check all words appear
    return all(w in text_lower for w in words)


def normalize_names(names: Set[str]) -> Set[str]:
    """
    Stage 3b: Normalize consultant names.

    Merge abbreviations with full names (e.g., "FW Cook" -> "Frederic W. Cook & Co.").
    """
    if len(names) <= 1:
        return names

    names_list = list(names)
    to_remove = set()

    for short in names_list:
        for long in names_list:
            if short == long or len(short) >= len(long):
                continue
            if is_abbreviation(short, long):
                to_remove.add(short)
                break

    return names - to_remove


def is_abbreviation(short: str, long: str) -> bool:
    """Check if short name is an abbreviation of long name."""
    short_lower = short.lower()
    long_lower = long.lower()

    # Substring match
    if short_lower in long_lower:
        return True

    # Alphanumeric match
    short_alpha = re.sub(r'[^a-z0-9]', '', short_lower)
    long_alpha = re.sub(r'[^a-z0-9]', '', long_lower)
    if short_alpha == long_alpha:
        return True

    return False


BLACKLIST_PATTERNS = [
    'conference board', 'worldatwork', 'nacd', 'salary.com', 'glassdoor',
    'indeed', 'linkedin', 'payscale', 'bureau of labor', 'iss', 'glass lewis',
    'bloomberg', 'factset', 'magazine', 'journal', 'association',
]


def filter_blacklisted(names: Set[str]) -> Set[str]:
    """Remove non-consultant sources from SURV."""
    filtered = set()
    for name in names:
        name_lower = name.lower()
        if not any(pattern in name_lower for pattern in BLACKLIST_PATTERNS):
            filtered.add(name)
    return filtered


# =============================================================================
# Aggregation (Stage 4)
# =============================================================================

def aggregate_chunks(chunk_results: List[Dict[str, List[str]]],
                     raw_mode: bool = False) -> Dict[str, List[str]]:
    """
    Stage 4: Aggregate results across chunks.

    Combines extractions from multiple chunks, handling:
    - Deduplication
    - Name normalization
    - RET/SURV mutual exclusivity
    """
    ret_set = set()
    surv_set = set()

    for result in chunk_results:
        ret_set.update(result.get('RET', []))
        surv_set.update(result.get('SURV', []))

    # Remove NO_CONSULTANTS markers
    ret_set.discard('NO_CONSULTANTS')
    surv_set.discard('NO_CONSULTANTS')

    # Drop UNNAMED if real names exist
    if len(ret_set) > 1 and 'UNNAMED' in ret_set:
        ret_set.discard('UNNAMED')
    if len(surv_set) > 1 and 'UNNAMED' in surv_set:
        surv_set.discard('UNNAMED')

    # Normalize names
    all_names = ret_set | surv_set
    normalized = normalize_names(all_names)
    ret_set = ret_set & normalized
    surv_set = surv_set & normalized

    if raw_mode:
        return {
            'RET': sorted(ret_set) if ret_set else ['NO_CONSULTANTS'],
            'SURV': sorted(surv_set) if surv_set else ['NO_CONSULTANTS'],
        }

    # Enforce mutual exclusivity (RET takes precedence)
    overlap = ret_set & surv_set
    surv_set -= overlap

    # Filter blacklisted sources
    surv_set = filter_blacklisted(surv_set)

    return {
        'RET': sorted(ret_set) if ret_set else ['NO_CONSULTANTS'],
        'SURV': sorted(surv_set) if surv_set else ['NO_CONSULTANTS'],
    }


# =============================================================================
# Full Pipeline
# =============================================================================

def process_document(file_path: str, model, tokenizer,
                     instruction: str = "long", raw_mode: bool = False) -> Dict:
    """Process a single proxy statement through the full pipeline."""
    content = read_file(file_path)
    if not content:
        return None

    # Extract metadata
    metadata = extract_header_info(content)
    filing_year = metadata.get('year')

    # Stage 1: Retrieval and chunking
    excerpts = find_relevant_excerpts(content)
    if not excerpts:
        return {
            'metadata': metadata,
            'retained_consultants': ['NO_CONSULTANTS'],
            'survey_consultants': ['NO_CONSULTANTS'],
        }

    combined = "\n\n<<<EXCERPT_BREAK>>>\n\n".join(e['text'] for e in excerpts)
    chunks = create_chunks(combined, tokenizer)

    # Stage 2-3: Extract from each chunk
    chunk_results = []
    for chunk in chunks:
        result = extract_from_chunk(
            chunk, model, tokenizer,
            filing_year=filing_year,
            instruction=instruction,
            raw_mode=raw_mode
        )
        chunk_results.append(result)

    # Stage 4: Aggregate
    final = aggregate_chunks(chunk_results, raw_mode=raw_mode)

    return {
        'metadata': metadata,
        'retained_consultants': final['RET'],
        'survey_consultants': final['SURV'],
    }


def process_folder(folder_path: str, model, tokenizer,
                   instruction: str = "long", raw_mode: bool = False) -> Dict:
    """Process all proxy statements in a company folder."""
    results = []

    for root, dirs, files in os.walk(folder_path):
        if "DEF 14A" in root:
            for f in files:
                if f == "full-submission.txt":
                    file_path = os.path.join(root, f)
                    result = process_document(
                        file_path, model, tokenizer,
                        instruction=instruction, raw_mode=raw_mode
                    )
                    if result:
                        results.append(result)

    return results


# =============================================================================
# Model Loading
# =============================================================================

def load_model(adapter_path: str = None, model_size: str = "27b"):
    """Load model with optional LoRA adapter."""
    model_name = f"google/gemma-3-{model_size}-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    try:
        from transformers import Gemma3ForConditionalGeneration
        base_model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    except ImportError:
        from transformers import AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"Loaded LoRA adapter from {adapter_path}")
    else:
        model = base_model
        print("Using vanilla base model (no adapter)")

    return model, tokenizer


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compensation consultant extraction pipeline")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--vanilla", action="store_true", help="Use base model without adapter")
    parser.add_argument("--model", default="27b", choices=["12b", "27b"], help="Model size")
    parser.add_argument("--data", required=True, help="Path to eval data folder")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--instruction", default="long", choices=["short", "long"],
                        help="Instruction format")

    args = parser.parse_args()

    # Load model
    adapter = None if args.vanilla else args.adapter
    raw_mode = args.vanilla
    model, tokenizer = load_model(adapter, args.model)

    # Find company folders
    data_path = Path(args.data)
    companies = [d for d in data_path.iterdir() if d.is_dir()]

    # Process all companies
    all_results = {}
    for company_dir in companies:
        print(f"Processing {company_dir.name}...")
        results = process_folder(
            str(company_dir), model, tokenizer,
            instruction=args.instruction, raw_mode=raw_mode
        )

        # Organize by year
        by_year = {}
        for r in results:
            year = r['metadata'].get('year', 'Unknown')
            by_year[year] = {
                'RET': r['retained_consultants'],
                'SURV': r['survey_consultants'],
            }

        all_results[company_dir.name] = {
            'consultants_by_year': by_year,
            'detailed_results': results,
        }

    # Save results
    output_data = {
        'processing_datetime': datetime.now().isoformat(),
        'company_results': all_results,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
