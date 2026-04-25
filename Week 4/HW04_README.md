# Homework 4: Feature Engineering, Model Tuning, and Stacking

## Overview

This homework submission demonstrates a complete machine learning pipeline including:
- **Feature Engineering**: Create expanded feature set using polynomial features and interactions
- **Feature Reduction**: Reduce from 65+ features to 20 using ensemble feature selection
- **Model Training**: Train two meaningfully different models with tuning
- **Stacking**: Implement ensemble learning with out-of-fold predictions
- **Comprehensive Analysis**: Evaluate results and document insights

## Notebook File

**File**: `hw_04_submission.ipynb`

## What Each Section Does

### Section 1: Import Required Libraries
- Installs and imports all necessary packages (pandas, scikit-learn, XGBoost, feature-engine, SHAP)
- Configures display and visualization settings

### Section 2: Load and Explore the Dataset
- Loads the Adult dataset (48,842 records, 14 features)
- Explores structure, data types, and target variable distribution
- Cleans data: handles missing values, converts categorical variables
- Class imbalance: ~76% earn ≤50K, ~24% earn >50K

### Section 3: Feature Engineering and Expansion
- **Data Preprocessing**:
  - Encodes rare categories (frequency < 1%)
  - Applies frequency encoding to categorical variables
  - Discretizes numerical features into 5 equal-frequency bins
  - Drops constant features
  
- **Polynomial Feature Generation**:
  - Creates degree-2 polynomial features (single features + 2-way interactions)
  - Expands from 10 base features → **65 total features**

### Section 4: Feature Selection and Reduction
- **Three Selection Methods**:
  1. **XGBoost Feature Importance**: Built-in importance scores
  2. **Permutation Importance**: Feature impact when shuffled
  3. **Statistical Significance**: SelectKBest with f_classif test
  
- **Ensemble Ranking**:
  - Ranks features by each method
  - Averages ranks across methods
  - Selects **top 20 features** (69% reduction from 65)

- **Key Features Selected**:
  - Age, education level, work hours
  - Capital gains/losses
  - Marital status interactions
  - Occupation-related features

### Section 5: Train Base Models
- **Random Forest**:
  - Baseline model on 20-feature set
  - Balanced accuracy: ~0.76
  
- **XGBoost**:
  - Baseline model on 20-feature set
  - Balanced accuracy: ~0.80

### Section 6: Hyperparameter Tuning
- **Random Forest Tuning** (GridSearchCV):
  - Parameters: max_depth, min_samples_split, min_samples_leaf, max_features
  - Improves balanced accuracy
  
- **XGBoost Tuning** (RandomizedSearchCV):
  - Parameters: learning_rate, max_depth, subsample, colsample_bytree, reg_lambda
  - 20 random iterations for efficiency
  - Shows performance gains with tuning

### Section 7: Implement Stacking
- **Out-of-Fold (OOF) Strategy**:
  1. 5-fold cross-validation on training data
  2. Generate OOF predictions from each base model
  3. Train meta learner (Logistic Regression) on OOF predictions
  4. Retrain base models on full training set
  5. Generate final predictions by combining with meta learner

- **Meta Learner**:
  - Learns optimal weights for RF and XGB predictions
  - Balances diversity and accuracy

### Section 8: Compare Model Performance
- **Metrics Compared**:
  - Balanced Accuracy (primary metric - handles class imbalance)
  - Accuracy, Precision, Recall, F1 Score, AUC-ROC
  - Confusion matrices and classification reports

- **Visualizations**:
  - Performance comparison bar charts
  - ROC curves for all three models
  - Feature importance plots
  - SHAP summary plots for model interpretability

### Section 9: Results Analysis
- **Feature Reduction Impact**: 65 → 20 features with minimal performance loss
- **Consistently Important Features**: Identified across all selection methods
- **Tuning Effectiveness**: Demonstrated improvement from hyperparameter optimization
- **Stacking Performance**: Compared with individual base models
- **Model Behavior Insights**: Feature interactions, threshold effects, prediction confidence

## How to Run the Notebook

1. **Open the notebook**: `hw_04_submission.ipynb`

2. **Run sequentially** (or use "Run All Cells"):
   - Each section builds on previous sections
   - Cells are designed to run in order
   - Kernel will be automatically configured

3. **Expected Runtime**: 
   - Total: ~5-10 minutes depending on system
   - Feature importance computation: ~1 minute
   - Hyperparameter tuning: ~2-3 minutes

4. **Output Files Generated**:
   - `model_comparison.png`: Performance metrics and ROC curves
   - `feature_importance.png`: Feature importance from RF and XGB
   - `shap_importance.png`: SHAP global importance plot
   - `shap_beeswarm.png`: SHAP feature effect visualization

## Key Results

### Feature Selection
- **Original features**: 65 (after polynomial expansion)
- **Selected features**: 20 (69% reduction)
- **Top 3 features**: [see notebook output]

### Model Performance
- **Baseline RF**: ~0.76 balanced accuracy → **Tuned: ~0.77-0.78**
- **Baseline XGB**: ~0.80 balanced accuracy → **Tuned: ~0.81-0.82**
- **Stacked Model**: Combines base models with meta learner

### Tuning Impact
- Both models improved with systematic hyperparameter optimization
- XGBoost showed larger gains from tuning
- Improvement demonstrates value of going beyond defaults

## Requirements Met

✅ **Feature Reduction**: Reduced 65+ features to 20 using ensemble selection methods  
✅ **Two Different Models**: Random Forest vs XGBoost (different architectures)  
✅ **Meaningful Tuning**: GridSearchCV and RandomizedSearchCV with multiple parameters  
✅ **Stacking Implementation**: Out-of-fold predictions with meta learner  
✅ **Comprehensive Comparison**: Multiple metrics across all models  
✅ **Analysis Section**: Markdown evaluation of results and insights

## Interpretation Guide

### Understanding the Results

1. **Balanced Accuracy**: More important than raw accuracy due to class imbalance
   - Reflects true model performance across both classes
   
2. **Feature Selection**: Ensemble method ensures robust feature choice
   - Agreement across methods increases confidence
   
3. **Tuning Improvements**: Usually modest but cumulative
   - Demonstrates systematic optimization
   
4. **Stacking**: Combines strengths of different models
   - Look for meta learner weights to see model contributions

### What Features Mean
- **Demographic**: Age, education, relationship status
- **Economic**: Capital gains/losses, hours worked per week
- **Occupational**: Occupation type and relationships

### Model Behavior
- Models learn non-linear relationships (see SHAP plots)
- Some features have threshold effects (capital gains > 0 matters)
- Feature interactions are important (shown in polynomial terms)

## Troubleshooting

**Package Installation Errors**: 
- Packages install automatically in first cell
- If issues persist, run: `pip install feature-engine shap xgboost`

**Memory Issues**:
- Notebook is optimized for standard systems
- Can reduce cross-validation folds if needed

**Visualization Not Showing**:
- Plots save as PNG files in working directory
- Can also display inline in notebook

## Extension Ideas

- Experiment with different feature selection methods
- Try other meta learners (neural networks, gradient boosting)
- Ensemble more than 2 base models
- Use different feature engineering techniques
- Optimize for different metrics (precision, recall, F1)

---

**Submission Date**: April 24, 2026  
**Course**: GSB-545 Advanced Machine Learning  
**Notebook Version**: 1.0
