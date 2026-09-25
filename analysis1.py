import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, roc_auc_score, mean_squared_error,
                             r2_score, accuracy_score, confusion_matrix, roc_curve,
                             auc, precision_recall_curve, average_precision_score)
import warnings
warnings.filterwarnings('ignore')


# 1. LOAD AND PREPARE DATA




conn = sqlite3.connect('database.sqlite')
samples = pd.read_sql_query("SELECT * FROM sampledata15", conn)
results = pd.read_sql_query("SELECT * FROM resultsdata15", conn)
conn.close()


# Clean data
samples = samples.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
results = results.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


# Use 'mean' column for detection (ND/NP = Non-Detect, O/R/A = Detect)
results['pesticide_detected'] = (~results['mean'].isin(['ND', 'NP'])).astype(int)
results['concen_numeric'] = pd.to_numeric(results['concen'], errors='coerce')


print(f"   Total test results: {len(results):,}")
print(f"   Detection rate: {results['pesticide_detected'].mean():.2%}")
print(f"   Detected samples: {results['pesticide_detected'].sum():,}")


# Merge datasets
merged = results.merge(samples, on='sample_pk', how='left', suffixes=('_res', '_samp'))


# Features
merged['is_imported'] = (merged['origin'] == '2').astype(int)
merged['is_organic'] = (merged['claim'] == 'PO').astype(int)
merged['concentration_log'] = np.log1p(merged['concen_numeric'].fillna(0))


print(f"   Merged dataset: {merged.shape}")


# 2. EXPLORATORY DATA ANALYSIS




# Detection by commodity
detection_by_commodity = merged.groupby('commod_samp')['pesticide_detected'].agg(['mean', 'count'])
detection_by_commodity = detection_by_commodity[detection_by_commodity['count'] > 100]


# Detection by origin
detection_by_origin = merged.groupby('origin')['pesticide_detected'].mean()


# Detection by country
detection_by_country = merged.groupby('country')['pesticide_detected'].agg(['mean', 'count'])
detection_by_country = detection_by_country[detection_by_country['count'] > 50]


print(f"\n   Top 10 Commodities:")
print(detection_by_commodity.sort_values('mean', ascending=False).head(10))


print(f"\n   By Origin:")
for origin_code, rate in detection_by_origin.items():
   origin_name = {1: 'Domestic', 2: 'Imported', 3: 'Unknown'}.get(int(origin_code), 'Other')
   print(f"      {origin_name}: {rate:.2%}")


# Concentration statistics
detected = merged[merged['concen_numeric'] > 0]
print(f"\n   Concentration stats (n={len(detected):,}):")
if len(detected) > 0:
   print(detected['concen_numeric'].describe())


# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))


# Plot 1: Detection by commodity
top_commod = detection_by_commodity.sort_values('mean', ascending=False).head(15)
top_commod_pct = top_commod['mean'] * 100


top_commod_pct.plot(kind='barh', ax=axes[0,0], color='steelblue', edgecolor='black')
axes[0,0].set_title('Top 15 Commodities by Detection Rate (%)', fontweight='bold', fontsize=12)
axes[0,0].set_xlabel('Detection Rate (%)')
axes[0,0].grid(axis='x', alpha=0.3)
for i, (idx, val) in enumerate(top_commod_pct.items()):
   axes[0,0].text(val + 0.1, i, f'{val:.2f}%', va='center', fontsize=9)


# Plot 2: Domestic vs Imported
origin_data = detection_by_origin.copy() * 100
labels = ['Domestic', 'Imported', 'Unknown']
bars = axes[0,1].bar(range(len(origin_data)), origin_data.values,
                     color=['green', 'orange', 'gray'], edgecolor='black', width=0.6)
axes[0,1].set_xticks(range(len(origin_data)))
axes[0,1].set_xticklabels(labels, fontsize=11)
axes[0,1].set_title('Detection Rate by Origin', fontweight='bold', fontsize=12)
axes[0,1].set_ylabel('Detection Rate (%)', fontsize=11)
axes[0,1].set_ylim([0, max(origin_data.values) * 1.3])
axes[0,1].grid(axis='y', alpha=0.3)


for i, bar in enumerate(bars):
   height = bar.get_height()
   axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                  f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')


# Plot 3: Detection by pesticide
pesticide_detection = merged.groupby('pestcode')['pesticide_detected'].agg(['mean', 'count'])
pesticide_detection = pesticide_detection[pesticide_detection['count'] > 1000]
top_pest = pesticide_detection.sort_values('mean', ascending=False).head(15)
top_pest_pct = top_pest['mean'] * 100


top_pest_pct.plot(kind='barh', ax=axes[1,0], color='coral', edgecolor='black')
axes[1,0].set_title('Top 15 Pesticides by Detection Rate (%)', fontweight='bold', fontsize=12)
axes[1,0].set_xlabel('Detection Rate (%)')
axes[1,0].grid(axis='x', alpha=0.3)
for i, (idx, val) in enumerate(top_pest_pct.items()):
   axes[1,0].text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)


# Plot 4: Detection by country
top_countries = detection_by_country.sort_values('mean', ascending=False).head(15)
top_countries_pct = top_countries['mean'] * 100


top_countries_pct.plot(kind='barh', ax=axes[1,1], color='purple', edgecolor='black')
axes[1,1].set_title('Top 15 Countries by Detection Rate (%)', fontweight='bold', fontsize=12)
axes[1,1].set_xlabel('Detection Rate (%)')
axes[1,1].grid(axis='x', alpha=0.3)
for i, (idx, val) in enumerate(top_countries_pct.items()):
   axes[1,1].text(val + 0.05, i, f'{val:.2f}%', va='center', fontsize=9)


plt.tight_layout()
plt.savefig('eda_final_fixed.png', dpi=300, bbox_inches='tight')




# 3. Predicting Detection




# Sample with BOTH classes (balanced sampling)
SAMPLE_SIZE = 25000
detected_samples = merged[merged['pesticide_detected'] == 1].sample(
   n=min(SAMPLE_SIZE, sum(merged['pesticide_detected'] == 1)), random_state=42
)
non_detected_samples = merged[merged['pesticide_detected'] == 0].sample(
   n=SAMPLE_SIZE, random_state=42
)
classification_sample = pd.concat([detected_samples, non_detected_samples])


print(f"   Sample: {len(classification_sample):,}")
print(f"   Detection rate: {classification_sample['pesticide_detected'].mean():.2%}")


# Encode features
le_commod = LabelEncoder()
le_pestcode = LabelEncoder()
le_country = LabelEncoder()
le_testclass = LabelEncoder()


classification_sample['commod_encoded'] = le_commod.fit_transform(
   classification_sample['commod_samp'].fillna('UNKNOWN')
)
classification_sample['pestcode_encoded'] = le_pestcode.fit_transform(
   classification_sample['pestcode'].fillna('UNKNOWN')
)
classification_sample['country_encoded'] = le_country.fit_transform(
   classification_sample['country'].fillna('UNKNOWN')
)
classification_sample['testclass_encoded'] = le_testclass.fit_transform(
   classification_sample['testclass'].fillna('UNKNOWN')
)


# Features
feature_cols = ['is_imported', 'is_organic', 'commod_encoded', 'pestcode_encoded',
               'country_encoded', 'testclass_encoded']


X_clf = classification_sample[feature_cols].fillna(0)
y_clf = classification_sample['pesticide_detected']


# Train-test split
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
   X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)


# Standardize
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)


print(f"   Train: {len(X_train_clf):,} | Test: {len(X_test_clf):,}")
print(f"   Train detection: {y_train_clf.mean():.2%}")


# Models
classification_models = {
   'Logistic Regression': {
       'model': LogisticRegression(max_iter=1000, random_state=42),
       'scaled': True
   },
   'LDA': {
       'model': LinearDiscriminantAnalysis(),
       'scaled': True
   },
   'QDA': {
       'model': QuadraticDiscriminantAnalysis(),
       'scaled': True
   },
   'Naive Bayes': {
       'model': GaussianNB(),
       'scaled': True
   },
   'Decision Tree': {
       'model': DecisionTreeClassifier(max_depth=10, random_state=42),
       'scaled': False
   },
   'Random Forest': {
       'model': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
       'scaled': False
   },
   'Extra Trees': {
       'model': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
       'scaled': False
   },
   'AdaBoost': {
       'model': AdaBoostClassifier(n_estimators=100, random_state=42),
       'scaled': False
   },
   'Gradient Boosting': {
       'model': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
       'scaled': False
   },
   'Neural Net': {
       'model': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
       'scaled': True
   }
}


classification_results = {}


for name, config in classification_models.items():
   print(f"\n   {name}...")
  
   model = config['model']
   use_scaled = config['scaled']
  
   try:
       if use_scaled:
           model.fit(X_train_clf_scaled, y_train_clf)
           y_pred = model.predict(X_test_clf_scaled)
           y_pred_proba = model.predict_proba(X_test_clf_scaled)[:, 1]
           cv_scores = cross_val_score(model, X_train_clf_scaled, y_train_clf,
                                      cv=5, scoring='roc_auc', n_jobs=-1)
       else:
           model.fit(X_train_clf, y_train_clf)
           y_pred = model.predict(X_test_clf)
           y_pred_proba = model.predict_proba(X_test_clf)[:, 1]
           cv_scores = cross_val_score(model, X_train_clf, y_train_clf,
                                      cv=5, scoring='roc_auc', n_jobs=-1)
      
       test_auc = roc_auc_score(y_test_clf, y_pred_proba)
       test_acc = accuracy_score(y_test_clf, y_pred)
      
       classification_results[name] = {
           'CV AUC': cv_scores.mean(),
           'CV Std': cv_scores.std(),
           'Test AUC': test_auc,
           'Test Accuracy': test_acc
       }
      




       print(f"      CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f}) | Test: {test_auc:.4f}")
      
   except Exception as e:
       print(f"      Error: {e}")
       classification_results[name] = {
           'CV AUC': np.nan, 'CV Std': np.nan, 'Test AUC': np.nan, 'Test Accuracy': np.nan
       }


# Results


clf_results_df = pd.DataFrame(classification_results).T.sort_values('Test AUC', ascending=False)
print(clf_results_df)
clf_results_df.to_csv('classification_results.csv')


# 4. DETAILED ANALYSIS OF BEST MODEL




# Get best model
best_model_name = clf_results_df['Test AUC'].idxmax()
best_model = classification_models[best_model_name]['model']
use_scaled = classification_models[best_model_name]['scaled']


print(f"   Best Model: {best_model_name}")
print(f"   Test AUC: {clf_results_df.loc[best_model_name, 'Test AUC']:.4f}")


# Predictions
X_data = X_test_clf_scaled if use_scaled else X_test_clf
y_pred = best_model.predict(X_data)
y_pred_proba = best_model.predict_proba(X_data)[:, 1]


# Confusion Matrix
cm = confusion_matrix(y_test_clf, y_pred)
print(f"\n   Confusion Matrix:")
print(f"   True Negatives:  {cm[0,0]:,}")
print(f"   False Positives: {cm[0,1]:,}")
print(f"   False Negatives: {cm[1,0]:,}")
print(f"   True Positives:  {cm[1,1]:,}")


accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


print(f"\n   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1 Score:  {f1:.4f}")


# Feature Importance
if hasattr(best_model, 'feature_importances_'):
   feat_imp = pd.DataFrame({
       'Feature': feature_cols,
       'Importance': best_model.feature_importances_
   }).sort_values('Importance', ascending=False)
  
   print(f"\n   Feature Importance:")
   print(feat_imp.to_string(index=False))
   feat_imp.to_csv('feature_importance.csv', index=False)


# 5. ADVANCED VISUALIZATIONS




fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)


# Plot 1: Model Compariso
ax1 = fig.add_subplot(gs[0, :])
clf_sorted = clf_results_df.sort_values('Test AUC', ascending=True)
colors = ['darkgreen' if x > 0.9 else 'green' if x > 0.75 else 'orange' if x > 0.65 else 'red'
         for x in clf_sorted['Test AUC']]
bars = ax1.barh(range(len(clf_sorted)), clf_sorted['Test AUC'], color=colors, edgecolor='black', alpha=0.8)
ax1.set_yticks(range(len(clf_sorted)))
ax1.set_yticklabels(clf_sorted.index, fontsize=10)
ax1.set_xlabel('Test AUC Score', fontsize=12, fontweight='bold')
ax1.set_title('Classification Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.axvline(x=0.9, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7, label='Excellent (>0.9)')
ax1.axvline(x=0.75, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Good (>0.75)')
ax1.legend(loc='lower right')
ax1.grid(axis='x', alpha=0.3)


for i, (idx, row) in enumerate(clf_sorted.iterrows()):
   ax1.text(row['Test AUC'] + 0.01, i, f"{row['Test AUC']:.3f}",
            va='center', fontsize=9, fontweight='bold')


# Plot 2: Confusion Matrix
ax2 = fig.add_subplot(gs[1, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False,
           xticklabels=['No Detection', 'Detection'],
           yticklabels=['No Detection', 'Detection'])
ax2.set_xlabel('Predicted', fontweight='bold')
ax2.set_ylabel('Actual', fontweight='bold')
ax2.set_title(f'Confusion Matrix\n{best_model_name}', fontweight='bold', fontsize=11)


# Plot 3: ROC Curve
ax3 = fig.add_subplot(gs[1, 1])
fpr, tpr, _ = roc_curve(y_test_clf, y_pred_proba)
roc_auc = auc(fpr, tpr)
ax3.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
ax3.set_xlabel('False Positive Rate', fontweight='bold')
ax3.set_ylabel('True Positive Rate', fontweight='bold')
ax3.set_title('ROC Curve', fontweight='bold', fontsize=11)
ax3.legend(loc='lower right')
ax3.grid(alpha=0.3)


# Plot 4: PrecisionRecall Curve
ax4 = fig.add_subplot(gs[1, 2])
precision_vals, recall_vals, _ = precision_recall_curve(y_test_clf, y_pred_proba)
avg_precision = average_precision_score(y_test_clf, y_pred_proba)
ax4.plot(recall_vals, precision_vals, color='darkred', lw=2,
        label=f'PR (AP = {avg_precision:.3f})')
ax4.set_xlabel('Recall', fontweight='bold')
ax4.set_ylabel('Precision', fontweight='bold')
ax4.set_title('Precision-Recall Curve', fontweight='bold', fontsize=11)
ax4.legend(loc='lower left')
ax4.grid(alpha=0.3)


# Plot 5: Feature Importance
ax5 = fig.add_subplot(gs[2, :])
if hasattr(best_model, 'feature_importances_'):
   feat_imp_sorted = feat_imp.sort_values('Importance', ascending=True)
   colors_feat = plt.cm.viridis(feat_imp_sorted['Importance'] / feat_imp_sorted['Importance'].max())
   bars = ax5.barh(range(len(feat_imp_sorted)), feat_imp_sorted['Importance'],
                   color=colors_feat, edgecolor='black')
   ax5.set_yticks(range(len(feat_imp_sorted)))
   ax5.set_yticklabels(feat_imp_sorted['Feature'], fontsize=10)
   ax5.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
   ax5.set_title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
   ax5.grid(axis='x', alpha=0.3)
  
   for i, (idx, row) in enumerate(feat_imp_sorted.iterrows()):
       ax5.text(row['Importance'] + 0.001, i, f"{row['Importance']:.3f}",
                va='center', fontsize=9)


plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')


# 6. MODEL CATEGORY ANALYSIS




model_categories = {
   'Linear Models': ['Logistic Regression', 'LDA', 'QDA', 'Naive Bayes'],
   'Tree-Based Models': ['Decision Tree', 'Random Forest', 'Extra Trees'],
   'Boosting Models': ['AdaBoost', 'Gradient Boosting'],
   'Neural Networks': ['Neural Net']
}


category_performance = {}
for category, models in model_categories.items():
   models_in_category = [m for m in models if m in clf_results_df.index]
   if models_in_category:
       avg_auc = clf_results_df.loc[models_in_category, 'Test AUC'].mean()
       max_auc = clf_results_df.loc[models_in_category, 'Test AUC'].max()
       category_performance[category] = {
           'Average AUC': avg_auc,
           'Best AUC': max_auc,
           'Best Model': clf_results_df.loc[models_in_category, 'Test AUC'].idxmax()
       }


category_df = pd.DataFrame(category_performance).T
print("\n   Model Category Performance:")
print(category_df.to_string())
category_df.to_csv('category_performance.csv')




# FINAL SUMMARY




summary_stats = {
   'Dataset Size': f"{len(merged):,} samples",
   'Detection Rate': f"{merged['pesticide_detected'].mean():.2%}",
   'Samples Analyzed': f"{len(classification_sample):,}",
   'Train/Test Split': '80/20',
   '': '',
   'Best Overall Model': best_model_name,
   'Best Test AUC': f"{clf_results_df['Test AUC'].max():.4f}",
   'Best CV AUC': f"{clf_results_df['CV AUC'].max():.4f}",
   ' ': '',
   'Model Performance': '',
   '  - Accuracy': f"{accuracy:.4f}",
   '  - Precision': f"{precision:.4f}",
   '  - Recall': f"{recall:.4f}",
   '  - F1 Score': f"{f1:.4f}",
   '  ': '',
   'Best Category': category_df['Best AUC'].idxmax(),
   'Category Best AUC': f"{category_df['Best AUC'].max():.4f}",
}


for key, val in summary_stats.items():
   if key == '':
       print()
   else:
       print(f"   {key}: {val}")


# summary saved
summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
summary_df.to_csv('final_summary.csv', index=False)








