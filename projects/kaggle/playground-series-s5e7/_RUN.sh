#!/bin/bash
# IMPORTANT SCRIPTS TO REPLICATE KEY FINDINGS
# ==========================================
# This file lists the most important scripts from the ml-manager/scripts directory
# that should be run to replicate the key findings for the personality prediction competition

# Set working directory
cd "$(dirname "$0")"

echo "=================================================="
echo "PERSONALITY PREDICTION - KEY SCRIPTS EXECUTION"
echo "=================================================="
echo ""

# 1. INITIAL ANALYSIS AND BASELINE
echo "1. INITIAL DATA ANALYSIS"
echo "------------------------"
echo "Run: python 20250703_1824_analyze_column_types.py"
echo "Purpose: Analyze data types and feature distributions"
echo ""

# 2. KEY DISCOVERY: 2.43% AMBIGUOUS PATTERN
echo "2. AMBIGUOUS PATTERN DISCOVERY"
echo "-------------------------------"
echo "Run: python 20250704_0007_ambivert_detector.py"
echo "Purpose: Discovers the 2.43% ambiguous personality pattern (ISFJs/ESFJs)"
echo ""

echo "Run: python 20250704_0010_detect_ambiverts_strategy.py"
echo "Purpose: Refines ambiguous detection strategy with 96.2% extrovert rule"
echo ""

# 3. MBTI MAPPING INSIGHTS
echo "3. MBTI TYPE MAPPING"
echo "--------------------"
echo "Run: python 20250704_0014_mbti_mapping_strategy.py"
echo "Purpose: Maps MBTI types to Introvert/Extrovert labels"
echo ""

echo "Run: python 20250704_0018_detect_mbti_mapping.py"
echo "Purpose: Detects specific MBTI patterns in the data"
echo ""

# 4. BASELINE ML MODELS
echo "4. BASELINE MODELS"
echo "------------------"
echo "Run: python 20250703_1831_xgboost_comparison.py"
echo "Purpose: XGBoost baseline without optimization"
echo ""

echo "Run: python 20250703_1838_xgboost_optuna_comparison.py"
echo "Purpose: XGBoost with Optuna hyperparameter optimization"
echo ""

echo "Run: python 20250703_1845_autogluon_comparison.py"
echo "Purpose: AutoGluon baseline for comparison"
echo ""

# 5. FEATURE ENGINEERING AND SELECTION
echo "5. FEATURE ENGINEERING"
echo "----------------------"
echo "Run: python 20250703_1854_xgboost_feature_engineering.py"
echo "Purpose: Creates engineered features for better performance"
echo ""

echo "Run: python 20250703_1932_feature_selection_comprehensive.py"
echo "Purpose: Comprehensive feature selection analysis"
echo ""

# 6. ADVANCED OPTIMIZATION
echo "6. ADVANCED STRATEGIES"
echo "----------------------"
echo "Run: python 20250703_2037_xgboost_ultimate_optimizer.py"
echo "Purpose: Ultimate XGBoost optimization with all tricks"
echo ""

echo "Run: python 20250704_0211_optimize_ambiguous_detection.py"
echo "Purpose: Optimize ambiguous detection thresholds with Optuna"
echo ""

# 7. IMPUTATION METHODS
echo "7. IMPUTATION STRATEGIES"
echo "------------------------"
echo "Run: python 20250704_0103_perfect_ml_imputation.py"
echo "Purpose: ML-based imputation for missing values"
echo ""

echo "Run: python 20250704_0110_compare_imputation_methods.py"
echo "Purpose: Compare different imputation approaches"
echo ""

# 8. PERFECT SCORE ATTEMPTS
echo "8. PERFECT SCORE STRATEGIES"
echo "---------------------------"
echo "Run: python 20250704_0049_perfect_duplicate_finder.py"
echo "Purpose: Find exact duplicates between train and test"
echo ""

echo "Run: python 20250704_0050_perfect_deterministic_seed.py"
echo "Purpose: Check for deterministic patterns in data generation"
echo ""

echo "Run: python 20250704_0055_perfect_gradient_boosting.py"
echo "Purpose: Gradient Boosting with 2.43% correction"
echo ""

# 9. FINAL SUBMISSIONS
echo "9. SUBMISSION GENERATION"
echo "------------------------"
echo "Run: python 20250703_2030_generate_submissions.py"
echo "Purpose: Generate various submission files"
echo ""

echo "Run: python 20250703_2125_generate_priority_submissions.py"
echo "Purpose: Generate priority submissions with best strategies"
echo ""

# 10. ADVANCED NEURAL STRATEGIES
echo "10. ADVANCED ML STRATEGIES"
echo "--------------------------"
echo "Run: python 20250704_1958_strategy_1_deep_mbti_reconstruction.py"
echo "Purpose: Deep learning approach to reconstruct MBTI types"
echo ""

echo "Run: python 20250704_2007_strategy_combined_breakthrough.py"
echo "Purpose: Combined breakthrough strategy using all insights"
echo ""

echo ""
echo "=================================================="
echo "EXECUTION NOTES:"
echo "=================================================="
echo "1. Ensure you're in the correct directory with data available"
echo "2. The key insight is the 2.43% ambiguous pattern (ISFJ/ESFJ types)"
echo "3. Best models achieve ~97.5% accuracy by handling ambiguous cases"
echo "4. The 'Drained_after_socializing' feature is highly predictive"
echo "5. Missing values in this feature often indicate ambiguous personalities"
echo ""
echo "RECOMMENDED EXECUTION ORDER:"
echo "1. Run analysis scripts (1-3) to understand the data"
echo "2. Run baseline models (4) to establish performance"
echo "3. Run optimization scripts (5-7) for best results"
echo "4. Run submission generation scripts for final predictions"
echo ""
echo "For quick results, focus on scripts from sections 2, 6, and 9."