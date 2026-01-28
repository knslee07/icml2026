#!/bin/bash
# Run evaluation for all model variants
# Usage: ./scripts/run_evals.sh

set -e

GOLD="data/gold_labels.json"
EVAL_DATA="data/eval/evals"
RESULTS_DIR="results"

mkdir -p $RESULTS_DIR

echo "========================================"
echo "Running evaluations..."
echo "========================================"

# 1. Domain-Specific (Long Instruction)
echo ""
echo "=== Domain-Specific (Long) ==="
python src/inference.py \
    --adapter adapters/domain_specific \
    --data $EVAL_DATA \
    --output $RESULTS_DIR/pred_domain_long.json \
    --instruction long

python src/evaluate.py \
    --predictions $RESULTS_DIR \
    --gold $GOLD \
    --output $RESULTS_DIR/eval_domain_long.csv \
    --normalize

# 2. Domain-Specific (Short Instruction)
echo ""
echo "=== Domain-Specific (Short) ==="
python src/inference.py \
    --adapter adapters/domain_specific \
    --data $EVAL_DATA \
    --output $RESULTS_DIR/pred_domain_short.json \
    --instruction short

python src/evaluate.py \
    --predictions $RESULTS_DIR \
    --gold $GOLD \
    --output $RESULTS_DIR/eval_domain_short.csv \
    --normalize

# 3. General IE (NER)
echo ""
echo "=== General IE (NER) ==="
python src/inference.py \
    --adapter adapters/ie_ner \
    --data $EVAL_DATA \
    --output $RESULTS_DIR/pred_ie_ner.json \
    --instruction long

python src/evaluate.py \
    --predictions $RESULTS_DIR \
    --gold $GOLD \
    --output $RESULTS_DIR/eval_ie_ner.csv \
    --normalize

# 4. General IE (Balanced)
echo ""
echo "=== General IE (Balanced) ==="
python src/inference.py \
    --adapter adapters/ie_balanced \
    --data $EVAL_DATA \
    --output $RESULTS_DIR/pred_ie_balanced.json \
    --instruction long

python src/evaluate.py \
    --predictions $RESULTS_DIR \
    --gold $GOLD \
    --output $RESULTS_DIR/eval_ie_balanced.csv \
    --normalize

# 5. Vanilla (Base Model)
echo ""
echo "=== Vanilla (Base Model) ==="
python src/inference.py \
    --vanilla \
    --data $EVAL_DATA \
    --output $RESULTS_DIR/pred_vanilla.json \
    --instruction long

python src/evaluate.py \
    --predictions $RESULTS_DIR \
    --gold $GOLD \
    --output $RESULTS_DIR/eval_vanilla.csv \
    --normalize

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "Results saved to: $RESULTS_DIR/"
echo "========================================"
