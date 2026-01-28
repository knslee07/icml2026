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
import ast
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# =============================================================================
# Configuration
# =============================================================================

# Retrieval settings
WINDOW_SIZE = 400          # Characters around keyword matches
MAX_CHUNK_TOKENS = 4000    # Tokens per LLM call
CHUNK_OVERLAP = 25         # Token overlap between chunks

# Keyword patterns for retrieval (hierarchical: specific -> general)
PATTERNS_SPECIFIC = [
    "compensation consult", "compensation advis",
    "independent consult", "independent advis",
    "committee consult", "committee advis",
    "outside consult", "outside advis",
    "proxy consult", "proxy advis",
    "external consult", "external advis",
    "third-party consult", "third-party advis",
    "remuneration consult", "remuneration advis",
    "benefit consult", "benefit advis",
    "benefits consult", "benefits advis",
]

PATTERNS_COMMITTEE = [
    "compensation committee", "benefits committee",
    "benefit committee", "remuneration committee",
    "HR Committee", "human resources committee",
    "human resource committee", "HRC ",
    "nominating committee", "nomination committee",
    "governance committee",
]

PATTERNS_GENERAL = [
    "executive compensation",
]

ALL_PATTERNS = [PATTERNS_SPECIFIC, PATTERNS_COMMITTEE, PATTERNS_GENERAL]

# Instruction prompts
SHORT_INSTRUCTION = """You are a meticulous reader of proxy-statement excerpts.

TASK: Identify two categories of compensation consultants:
  1. RET: Consultants retained/engaged to advise on executive compensation (includes advisory, analysis, market studies, recommendations).
  2. SURV: Survey/data providers NOT explicitly retained as advisors (includes proprietary surveys like Radford, Mercer, Towers Watson).

RULES:
- RET and SURV are mutually exclusive
- If consultant replaced during the period, list ONLY the final consultant
- Exclude: directors-only consultants, law firms, internal HR staff, public data sources (Salary.com, BLS, proxy advisors)
- If name undisclosed: 'UNNAMED'. If none found: 'NO_CONSULTANTS'

INPUT: Excerpts separated by <<<EXCERPT_BREAK>>>. Treat as one document.

OUTPUT FORMAT:
{RET: 'Firm A', 'Firm B'}, {SURV: 'Firm C'}
- Use longest name variant
- Alphabetical order, single quotes

TIMING: Filing year (shown at document start) OR prior year only. If consultant replaced, list only final consultant even if predecessor worked during this window.
"""

LONG_INSTRUCTION = """You are a meticulous reader of proxy-statement excerpts.

TASK: Identify two categories of compensation consultants:
  1. RET: Consultants retained/engaged by committee/management/Board to advise on executive compensation.
     - Includes: advisory, analysis, design, market studies, recommendations, reviews (even without recommendations), SERP/retirement benefits consultants.
     - Engagement structure (retainer vs. case-by-case) does NOT matter.
  2. SURV: Survey/data providers for executive compensation NOT explicitly retained as advisors.
     - Includes: proprietary surveys (Radford, Mercer, Towers Watson), data aggregators, HR/internal-accessed survey data (if USED for exec comp, label provider as SURV).
     - RET and SURV are mutually exclusive.

CRITICAL SPECIAL CASES:
- Consultant replacements: List ONLY FINAL consultant, NOT predecessor. Watch for: 'going forward', 'transitioned to', 'selected as new', 'reevaluated', 'in connection with [event]'. List NOT exhaustive\u2014read for any replacement indication. Even if predecessor worked during temporal window, exclude if replaced.

- Director compensation only: Exclude (NO_CONSULTANTS for both RET/SURV)
- UNNAMED: If retained but name undisclosed
- Fee amounts: IRRELEVANT (RET if retained regardless of fee)
- Internal teams/employees: Exclude HR staff, executives (CSO, CFO, CEO)
- Subsidiaries: Exclude consultants engaged by subsidiaries
- Law firms: Exclude legal advisors (Skadden, Wilson Sonsini, Sullivan & Cromwell, Wachtell Lipton)
- Sub-consultants: Exclude pension actuaries
- Peer companies: NOT SURV providers
- Merger/acquisition: Only current company's post-merger consultants

EXCLUSIONS - Public Data Sources (NO_CONSULTANTS for both RET/SURV):
- Public websites: Salary.com, Payscale, Glassdoor, Indeed, LinkedIn Salary, Levels.fyi
- Government: BLS, SEC filings
- Database aggregators without consultant: Bloomberg, FactSet, ISS downloads
- Proxy advisors: ISS, Glass Lewis
- Research orgs: Conference Board, WorldAtWork, NACD
- Direct sources: Peer proxy statements
\u26a0\ufe0f Only include proprietary surveys from compensation consulting firms.

OTHER RULES:
- Survey products: firm name only ('Radford Survey' \u2192 'Radford')
- Individual names: resolve to firm
- Historical relationships: assume current if implied ('worked 15 years')
- RFP (Request for Proposal) without selection: NO_CONSULTANTS
- Data licensing without consultant: NO_CONSULTANTS

INPUT FORMAT:
Excerpts separated by <<<EXCERPT_BREAK>>> or '=== LLM CHUNK k/n ==='. Treat as one document.

OUTPUT RULES (CUSTOM FORMAT; NOT JSON):
1. Return: {RET: 'Firm A', 'Firm B'}, {SURV: 'Firm C'}
2. Multiple names: use LONGEST name
3. Survey products: firm name only (Mercer Benchmark \u2192 Mercer)
4. Alphabetical order, single quotes
5. None found: 'NO_CONSULTANTS'. Name undisclosed: 'UNNAMED'

TIMING RULES:
- Temporal window: filing year OR prior year
- Filing year: at document start (e.g., 'filed in 2023')
- No year info: assume current unless indicated otherwise
- Fiscal year: May differ from calendar (proxy filed 2024 for FY2023 may cover Apr 2023-Mar 2024). Filing/prior year typically coincides with fiscal year, but fiscal info may not be available.
- CRITICAL: If consultant replaced, list ONLY final consultant even if predecessor worked during temporal window
- Historical data: Consultants from BEFORE temporal window without current retention indication = NO_CONSULTANTS
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
    """Clean HTML entities and normalize whitespace (character-by-character)."""
    text = html.unescape(text)
    cleaned = []
    in_tag = False
    last_was_space = True

    for char in text:
        if char == '<':
            in_tag = True
            continue
        elif char == '>':
            in_tag = False
            continue
        elif in_tag:
            continue

        is_space = char.isspace()
        if is_space:
            if last_was_space:
                continue
            char = ' '
            last_was_space = True
        else:
            last_was_space = False

        cleaned.append(char)

    return ''.join(cleaned)


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
    Merging is done per pattern group to preserve group boundaries.
    """
    cleaned = clean_text(text)
    merged_excerpts = []

    for pattern_group in ALL_PATTERNS:
        hits = []
        for pattern in pattern_group:
            for match in re.finditer(pattern, cleaned, re.IGNORECASE):
                start = max(0, match.start() - window)
                end = min(len(cleaned), match.end() + window)
                hits.append({
                    'text': cleaned[start:end],
                    'position': start,
                    'pattern': pattern
                })
        merged_excerpts += merge_overlapping(hits)

    return merged_excerpts


def merge_overlapping(excerpts: List[Dict]) -> List[Dict]:
    """Merge overlapping text excerpts."""
    if not excerpts:
        return []

    sorted_exc = sorted(excerpts, key=lambda x: x['position'])
    merged = [sorted_exc[0].copy()]

    for exc in sorted_exc[1:]:
        last = merged[-1]
        last_end = last['position'] + len(last['text'])

        if exc['position'] <= last_end:
            if exc['position'] + len(exc['text']) > last_end:
                extension = exc['text'][last_end - exc['position']:]
                last['text'] += extension
            if exc['pattern'] != last['pattern']:
                last['pattern'] = f"{last['pattern']}, {exc['pattern']}"
        else:
            merged.append(exc.copy())

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

def _extract_names_with_quotes(raw: str) -> List[str]:
    """Extract single-quoted names using state machine.

    Handles possessives (Cook's) and edge cases.
    """
    names = []
    in_quote = False
    current_name = ""
    i = 0

    while i < len(raw):
        char = raw[i]

        if char == "'" and not in_quote:
            in_quote = True
            current_name = ""
        elif char == "'" and in_quote:
            if i + 1 < len(raw) and raw[i + 1] == 's':
                current_name += char
            else:
                in_quote = False
                if current_name.strip():
                    names.append(current_name.strip())
                current_name = ""
        elif in_quote:
            current_name += char

        i += 1

    return names


def parse_llm_response(response: str, chunk_id: str = "") -> Dict[str, List[str]]:
    """Parse LLM output into RET and SURV lists.

    Multiple parsing paths for robustness:
    - PATH 0: Python dict (ast.literal_eval)
    - PATH 1: Strict regex {RET: ...}, {SURV: ...}
    - PATH 2: Relaxed regex (same braces)
    - PATH 2.5: Bracketed lists
    - PATH 3: Fallback (one name per line)
    """
    # Clean response
    stop_words = ["model", "ANSWER:", "<END>"]
    for stop_word in stop_words:
        if stop_word in response:
            response = response.split(stop_word)[0].strip()
            break
    response = response.replace("<END>", "").strip()

    # Handle code block format
    code_match = re.search(r'```(?:python)?\s*(.*?)```', response, re.DOTALL)
    if code_match:
        response = code_match.group(1).strip()

    # PATH 0: Try ast.literal_eval for Python dict
    try:
        parsed = ast.literal_eval(response.strip())
        if isinstance(parsed, dict) and 'RET' in parsed and 'SURV' in parsed:
            ret = parsed['RET'] if isinstance(parsed['RET'], list) else [parsed['RET']]
            surv = parsed['SURV'] if isinstance(parsed['SURV'], list) else [parsed['SURV']]
            ret = [r for r in ret if r and r != 'NO_CONSULTANTS']
            surv = [s for s in surv if s and s != 'NO_CONSULTANTS']
            return {"RET": sorted(set(ret)), "SURV": sorted(set(surv))}
    except (ValueError, SyntaxError):
        pass

    # PATH 1: Strict regex {RET: ...}, {SURV: ...}
    patterns = [
        r"\{RET:\s*([^}]*)\},\s*\{SURV:\s*([^}]*)\}",
        r"\{'RET':\s*([^}]*)\},\s*\{'SURV':\s*([^}]*)\}",
        r'\{"RET":\s*([^}]*)\},\s*\{"SURV":\s*([^}]*)\}',
    ]
    for pattern in patterns:
        m = re.search(pattern, response, re.I)
        if m:
            ret = _extract_names_with_quotes(m.group(1))
            surv = _extract_names_with_quotes(m.group(2))
            return {"RET": sorted(set(ret)), "SURV": sorted(set(surv))}

    # PATH 2: Relaxed regex (same braces)
    m2 = re.search(r"\{[^{}]*RET:\s*(.*?)\s*,?\s*SURV:\s*(.*?)\s*\}", response, flags=re.I | re.S)
    if m2:
        ret = _extract_names_with_quotes(m2.group(1))
        surv = _extract_names_with_quotes(m2.group(2))
        return {"RET": sorted(set(ret)), "SURV": sorted(set(surv))}

    # PATH 2.5: Bracketed lists
    lists_found = re.findall(r'\[(.*?)\]', response, re.S)
    if lists_found:
        all_names = []
        for list_content in lists_found:
            names = _extract_names_with_quotes(list_content)
            all_names.extend(names)
        if all_names:
            return {"RET": sorted(set(all_names)), "SURV": []}

    # PATH 3: Fallback - one name per line
    allowed = re.compile(r"^[A-Za-z0-9&., '\\-]+$")
    names = [ln.strip() for ln in response.splitlines()
             if ln.strip() and allowed.match(ln) and not ln.startswith('<')]
    return {"RET": sorted(set(names)), "SURV": []}


def extract_from_chunk(text: str, model, tokenizer, filing_year: str = None,
                       instruction: str = "long", chunk_id: str = "",
                       raw_mode: bool = False,
                       source_text: str = None) -> Dict[str, List[str]]:
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
            f"TEMPORAL CONTEXT: This document was filed in {filing_year}. "
            f"Consultants retained in {filing_year} or {int(filing_year)-1} are considered 'currently retained'.\n"
        )
    prompt_parts.append(f"TEXT TO ANALYZE: {text}")
    prompt_parts.append(
        "After reading the excerpts, output ONLY one line in this exact format:\n"
        "{RET: 'Firm A', 'Firm B'}, {SURV: 'Firm C'}\n"
        "If no consultants appear anywhere, output exactly:\n"
        "{RET: 'NO_CONSULTANTS'}, {SURV: 'NO_CONSULTANTS'}\n"
        "Do not summarize or repeat the excerpts.\n"
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
            early_stopping=True,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Parse response
    result = parse_llm_response(response, chunk_id=chunk_id)

    # Validate against source text (skip for raw/vanilla mode)
    validation_text = source_text if source_text else text
    if not raw_mode:
        result = validate_against_text(result, validation_text)

    return result


# =============================================================================
# Post-Processing (Stage 3)
# =============================================================================

def find_firm_in_text(firm: str, text: str) -> str:
    """Validate that a consultant name appears in the source text.

    Uses word-by-word sequence matching with gap tolerance.
    Returns the matched name from text, or None if not found.
    """
    firm_lower = firm.lower()

    # Tokenize firm name preserving & and .
    firm_words = []
    current_word = ""
    for char in firm_lower:
        if char.isalnum() or char in ['&', '.']:
            current_word += char
        else:
            if current_word:
                firm_words.append(current_word)
                current_word = ""
    if current_word:
        firm_words.append(current_word)

    # Filter short words
    firm_words = [w for w in firm_words if len(w) > 1 or (len(w) == 1 and w.isalpha()) or '.' in w]

    if not firm_words:
        return None

    # Find all positions of each word in text
    text_lower = text.lower()
    word_positions = []
    for word in firm_words:
        positions = []
        start = 0
        while True:
            pos = text_lower.find(word, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        if not positions:
            return None
        word_positions.append(positions)

    if not word_positions or not word_positions[0]:
        return None

    # Find words appearing in sequence with max gap
    max_gap = 5  # Max words between consecutive firm words

    for first_pos in word_positions[0]:
        current_pos = first_pos
        found_sequence = [current_pos]

        success = True
        for next_word_positions in word_positions[1:]:
            valid_next_pos = None
            for pos in next_word_positions:
                if 0 < pos - current_pos < max_gap * 15:
                    valid_next_pos = pos
                    break

            if valid_next_pos is None:
                success = False
                break

            found_sequence.append(valid_next_pos)
            current_pos = valid_next_pos

        if not success:
            continue

        if len(found_sequence) == len(firm_words):
            start_pos = found_sequence[0]
            end_pos = found_sequence[-1] + len(firm_words[-1])

            # Expand boundaries
            while start_pos > 0 and (text[start_pos-1].isalnum() or
                                     text[start_pos-1] in '&., '):
                start_pos -= 1
            while end_pos < len(text) and (text[end_pos].isalnum() or
                                           text[end_pos] in '&., '):
                end_pos += 1

            company_name = text[start_pos:end_pos].strip()
            if all(word in company_name.lower() for word in firm_words):
                return company_name.strip(' ,():;')

    return None


def validate_against_text(result: Dict[str, List[str]], source_text: str) -> Dict[str, List[str]]:
    """
    Stage 3a: Validate extracted names against source text.

    Reject hallucinations - names that don't appear in the source.
    """
    validated = {'RET': [], 'SURV': []}
    special_values = {'UNNAMED', 'NO_CONSULTANTS'}

    for category in ['RET', 'SURV']:
        for name in result.get(category, []):
            if name.upper() in special_values:
                validated[category].append(name)
            elif find_firm_in_text(name, source_text):
                validated[category].append(name)

    return validated


def _build_canonical_map(names: set) -> dict:
    """Build a mapping from abbreviations to their canonical (longest) names."""
    if len(names) <= 1:
        return {}

    canonical_map = {}
    names_list = list(names)

    for i, short in enumerate(names_list):
        for j, long in enumerate(names_list):
            if i == j:
                continue
            if len(short) >= len(long):
                continue

            if _is_abbreviation_of(short, long):
                canonical_map[short] = long
                break

    return canonical_map


def _is_abbreviation_of(short: str, long: str) -> bool:
    """Check if short name is an abbreviation of long name."""
    short_lower = short.lower().strip()
    long_lower = long.lower().strip()

    # Case 0: Typo/spacing variants
    def normalize_spacing(s):
        s = re.sub(r'\s*&\s*', ' & ', s)
        s = re.sub(r'\s*,\s*', ', ', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    if normalize_spacing(short_lower) == normalize_spacing(long_lower):
        return True

    # Case 1: Substring
    if short_lower in long_lower:
        return True

    # Case 2: Alphanumeric substring
    short_clean = re.sub(r'[^a-z0-9\s]', '', short_lower)
    long_clean = re.sub(r'[^a-z0-9\s]', '', long_lower)
    if short_clean and short_clean in long_clean:
        return True

    # Case 2b: Full alphanumeric match
    short_alphanum = re.sub(r'[^a-z0-9]', '', short_lower)
    long_alphanum = re.sub(r'[^a-z0-9]', '', long_lower)
    if short_alphanum == long_alphanum:
        return True

    # Case 3: Known aliases
    KNOWN_ALIASES = {
        'frederic w. cook': ['fwc', 'f.w.c', 'fw.c', 'fw cook', 'f.w. cook', 'cook & co', 'fred cook', 'frederick cook'],
        'pearl meyer': ['pm&p', 'pmp', 'pearl meyer & partners', 'pearl, meyer & partners'],
        'towers watson': ['tw', 'towers'],
        'willis towers watson': ['wtw', 'willis tw'],
        'hewitt associates': ['hewitt'],
        'aon hewitt': ['aon'],
        'compensia': ['compensia inc'],
        'pay governance': ['pay governance llc', 'pg'],
        'semler brossy': ['semler brossy consulting group', 'sbcg'],
        'meridian compensation partners': ['meridian', 'meridian cp', 'mcp'],
        'exequity': ['exequity llp'],
        'farient advisors': ['farient'],
        'mercer': ['mercer consulting', 'mercer human resource'],
        'radford': ['aon radford'],
        'mclagan': ['aon mclagan'],
        'korn ferry': ['korn ferry hay group', 'hay group', 'hay'],
        'buck consultants': ['buck'],
        'compensation advisory partners': ['cap'],
        'lyons benenson': ['lyons, benenson'],
    }

    for canonical, aliases in KNOWN_ALIASES.items():
        if canonical in long_lower:
            if short_lower in aliases or any(alias in short_lower for alias in aliases):
                return True

    # Case 4: Initials match
    long_words = re.findall(r'[A-Za-z]+', long)
    long_initials_all = ''.join(w[0].upper() for w in long_words)
    short_letters = re.sub(r'[^A-Za-z]', '', short).upper()

    if len(short_letters) >= 2 and long_initials_all.startswith(short_letters):
        return True

    # Case 4b: Initials excluding common suffixes
    suffix_words = {'co', 'inc', 'llc', 'llp', 'corp', 'ltd', 'group', 'partners', 'consulting', 'advisors', 'associates'}
    long_initials_no_suffix = ''.join(w[0].upper() for w in long_words if w.lower() not in suffix_words)
    if len(short_letters) >= 2 and short_letters == long_initials_no_suffix:
        return True

    # Case 5: First word match
    long_first_word = long_words[0].lower() if long_words else ''
    short_first_word = re.findall(r'[A-Za-z]+', short_lower)
    if short_first_word and short_first_word[0] == long_first_word and len(short) < len(long) / 2:
        return True

    return False


# Blacklist patterns for non-consultant sources
BLACKLIST_PATTERNS = [
    # Industry/professional associations
    'bankers association', 'bar association', 'medical association',
    'hospital association', 'insurance association', 'credit union',
    'chamber of commerce',
    # Trade publications
    'magazine', 'journal', 'director magazine', 'bank director',
    # Public data sources
    'conference board', 'worldatwork', 'nacd',
    'salary.com', 'glassdoor', 'indeed', 'linkedin', 'levels.fyi', 'payscale',
    'bureau of labor', 'bls',
    # Proxy advisors
    'iss', 'glass lewis',
    # Database aggregators
    'bloomberg', 'factset',
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
    - Name normalization (canonical mapping)
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

    # Build canonical name map and apply to both sets
    all_names = ret_set | surv_set
    canonical_map = _build_canonical_map(all_names)
    ret_set = {canonical_map.get(name, name) for name in ret_set}
    surv_set = {canonical_map.get(name, name) for name in surv_set}

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

    BREAK = "\n\n<<<EXCERPT_BREAK>>>\n\n"
    combined = BREAK.join(e['text'] for e in excerpts)
    chunks = create_chunks(combined, tokenizer)

    # Stage 2-3: Extract from each chunk
    chunk_results = []
    for idx, chunk in enumerate(chunks, 1):
        result = extract_from_chunk(
            chunk, model, tokenizer,
            filing_year=filing_year,
            instruction=instruction,
            chunk_id=f"{idx}/{len(chunks)}",
            raw_mode=raw_mode,
            source_text=chunk,
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
