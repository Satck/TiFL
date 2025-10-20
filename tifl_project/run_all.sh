#!/bin/bash

# TiFL完整实验批量运行脚本

DATASETS=("mnist" "fashion_mnist" "cifar10")
STRATEGIES=("vanilla" "uniform" "fast" "slow" "adaptive")

echo "=========================================="
echo "TiFL Complete Reproduction"
echo "=========================================="

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Dataset: $dataset"
    echo "------------------------------------------"
    
    for strategy in "${STRATEGIES[@]}"; do
        echo "Running $strategy..."
        python main.py --dataset $dataset --strategy $strategy
        
        if [ $? -eq 0 ]; then
            echo "✓ $dataset - $strategy completed"
        else
            echo "✗ $dataset - $strategy failed"
        fi
    done
    
    # 生成对比图
    echo "Generating plots for $dataset..."
    python visualize.py --dataset $dataset
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
