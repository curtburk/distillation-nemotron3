#!/bin/bash
cd ~/pmm-demos/knowledge-distillation-training
ps -p $(cat train.pid) &>/dev/null && echo "=== STILL TRAINING ===" || echo "=== COMPLETE OR DIED ==="
echo ""
echo "Loss trajectory (step | loss):"
grep "'loss':" logs/train.log | awk -F"'loss': '" '{print $2}' | awk -F"'" '{print NR"\t"$1}' | tail -30
echo ""
echo "Checkpoints:"
ls -la lora_adapter/ 2>/dev/null
