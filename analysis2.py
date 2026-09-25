import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')


# 1. LOAD DATA


conn = sqlite3.connect('database.sqlite')
samples = pd.read_sql_query("SELECT * FROM sampledata15", conn)
results = pd.read_sql_query("SELECT * FROM resultsdata15", conn)
conn.close()


samples = samples.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
results = results.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


results['pesticide_detected'] = (~results['mean'].isin(['ND', 'NP'])).astype(int)


merged = results.merge(samples, on='sample_pk', how='left', suffixes=('_res', '_samp'))


print(f"   Total samples: {results['sample_pk'].nunique():,}")
print(f"   Unique pesticides: {results['pestcode'].nunique()}")
print(f"   Unique commodities: {merged['commod_samp'].nunique()}")


# 2. COMMODITY-PESTICIDE MATRIX


detection_matrix = pd.crosstab(
   merged['commod_samp'],
   merged['pestcode'],
   values=merged['pesticide_detected'],
   aggfunc='mean'
).fillna(0)


frequency_matrix = pd.crosstab(
   merged['commod_samp'],
   merged['pestcode'],
   values=merged['pesticide_detected'],
   aggfunc='count'
).fillna(0)


print(f"   Matrix shape: {detection_matrix.shape}")
print(f"   (Commodities × Pesticides): {detection_matrix.shape[0]} × {detection_matrix.shape[1]}")


pesticides_to_keep = frequency_matrix.sum(axis=0) >= 100
detection_matrix_filtered = detection_matrix.loc[:, pesticides_to_keep]
frequency_matrix_filtered = frequency_matrix.loc[:, pesticides_to_keep]


print(f"   After filtering: {detection_matrix_filtered.shape[0]} × {detection_matrix_filtered.shape[1]}")


# 3. MATRIX FACTORIZATION


n_components = 20
nmf_model = NMF(n_components=n_components, init='random', random_state=42, max_iter=500)


W = nmf_model.fit_transform(detection_matrix_filtered) 
H = nmf_model.components_ 


predicted_detection = W @ H
predicted_detection_df = pd.DataFrame(
   predicted_detection,
   index=detection_matrix_filtered.index,
   columns=detection_matrix_filtered.columns
)


reconstruction_error = nmf_model.reconstruction_err_
print(f"   Model trained with {n_components} latent factors")
print(f"   Reconstruction error: {reconstruction_error:.4f}")
# 4. RECOMMENDATION FUNCTION




def recommend_pesticides(commodity, top_n=10, min_detection_rate=0.01):
   """
   Recommend top pesticides to test for a given commodity
   """
   if commodity not in predicted_detection_df.index:
       return None
  
   scores = predicted_detection_df.loc[commodity]
  
   scores = scores[scores >= min_detection_rate]
  
   top_pesticides = scores.sort_values(ascending=False).head(top_n)
  
   recommendations = pd.DataFrame({
       'Pesticide': top_pesticides.index,
       'Predicted_Detection_Rate': top_pesticides.values,
       'Actual_Detection_Rate': [detection_matrix_filtered.loc[commodity, p] if p in detection_matrix_filtered.columns else 0
                                  for p in top_pesticides.index],
       'Times_Tested': [frequency_matrix_filtered.loc[commodity, p] if p in frequency_matrix_filtered.columns else 0
                       for p in top_pesticides.index]
   }).reset_index(drop=True)
  
   return recommendations


def estimate_cost_savings(commodity, top_n=10):
   """
   Estimate cost savings by testing only top N pesticides
   """
   if commodity not in detection_matrix_filtered.index:
       return None
   total_tested = (frequency_matrix_filtered.loc[commodity] > 0).sum()
  
   recs = recommend_pesticides(commodity, top_n=top_n)
  
   actual_detections = detection_matrix_filtered.loc[commodity]
   top_pests = recs['Pesticide'].values
  
   coverage = actual_detections[top_pests].sum() / actual_detections[actual_detections > 0].sum() if actual_detections.sum() > 0 else 0
  
   tests_saved = total_tested - top_n
   cost_per_test = 50 
   savings_per_sample = tests_saved * cost_per_test
  
   return {
       'Total_Pesticides_Tested': total_tested,
       'Recommended_Pesticides': top_n,
       'Tests_Saved': tests_saved,
       'Detection_Coverage': f"{coverage:.1%}",
       'Cost_Savings_Per_Sample': f"${savings_per_sample:,.0f}",
       'Efficiency_Gain': f"{(tests_saved/total_tested)*100:.1f}%"
   }
# 5. RECOMMENDATIONS FOR TOP COMMODITIES


top_commodities = frequency_matrix_filtered.sum(axis=1).sort_values(ascending=False).head(10)


print(f"\n   Top 10 Commodities by Testing Volume:")
all_recommendations = {}
all_savings = {}


for commodity in top_commodities.index:
   recs = recommend_pesticides(commodity, top_n=10)
   savings = estimate_cost_savings(commodity, top_n=10)
  
   all_recommendations[commodity] = recs
   all_savings[commodity] = savings
  
   print(f"\n   {commodity}:")
   print(f"      Currently tests: {savings['Total_Pesticides_Tested']} pesticides")
   print(f"      Recommend: {savings['Recommended_Pesticides']} pesticides")
   print(f"      Detection coverage: {savings['Detection_Coverage']}")
   print(f"      Savings per sample: {savings['Cost_Savings_Per_Sample']}")
   print(f"      Top 5 recommended: {', '.join(recs['Pesticide'].head(5).values)}")




# 6. VISUALIZATIONS
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)


# Plot 1: Recommendation Heatmape
ax1 = fig.add_subplot(gs[0, :2])
top_10_comm = top_commodities.head(10).index
top_20_pests = detection_matrix_filtered.sum(axis=0).sort_values(ascending=False).head(20).index


heatmap_data = detection_matrix_filtered.loc[top_10_comm, top_20_pests] * 100


sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', cbar_kws={'label': 'Detection Rate (%)'},
           ax=ax1, linewidths=0.5)
ax1.set_title('Detection Rate Heatmap: Top 10 Commodities × Top 20 Pesticides',
             fontsize=14, fontweight='bold')
ax1.set_xlabel('Pesticide Code', fontsize=12, fontweight='bold')
ax1.set_ylabel('Commodity', fontsize=12, fontweight='bold')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')


# Plot 2: Cost Savings Potential
ax2 = fig.add_subplot(gs[0, 2])
savings_data = pd.DataFrame(all_savings).T
savings_data['Tests_Saved_Num'] = savings_data['Tests_Saved']
savings_data_sorted = savings_data.sort_values('Tests_Saved_Num', ascending=True)


ax2.barh(range(len(savings_data_sorted)), savings_data_sorted['Tests_Saved_Num'],
        color='green', edgecolor='black', alpha=0.7)
ax2.set_yticks(range(len(savings_data_sorted)))
ax2.set_yticklabels(savings_data_sorted.index, fontsize=10)
ax2.set_xlabel('Tests Saved per Sample', fontsize=11, fontweight='bold')
ax2.set_title('Cost Savings Potential\n(by Commodity)', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)


for i, val in enumerate(savings_data_sorted['Tests_Saved_Num']):
   ax2.text(val + 1, i, f'{int(val)}', va='center', fontsize=9, fontweight='bold')


# Plot 3: Detection Coverage
ax3 = fig.add_subplot(gs[1, 0])
coverage_values = [float(s['Detection_Coverage'].strip('%')) for s in all_savings.values()]
coverage_commodities = list(all_savings.keys())


bars = ax3.bar(range(len(coverage_values)), coverage_values, color='steelblue', edgecolor='black', alpha=0.7)
ax3.set_xticks(range(len(coverage_commodities)))
ax3.set_xticklabels(coverage_commodities, rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('Detection Coverage (%)', fontsize=11, fontweight='bold')
ax3.set_title('Detection Coverage with Top 10 Pesticides\n(vs. testing all)',
             fontsize=13, fontweight='bold')
ax3.axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% threshold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)


for bar, val in zip(bars, coverage_values):
   height = bar.get_height()
   ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
           f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')


# Plot 4: Ex - Apple Recommendations
ax4 = fig.add_subplot(gs[1, 1])
if 'AP' in all_recommendations:
   apple_recs = all_recommendations['AP'].head(10)
   ax4.barh(range(len(apple_recs)), apple_recs['Predicted_Detection_Rate'] * 100,
            color='coral', edgecolor='black', alpha=0.7, label='Predicted')
   ax4.scatter(apple_recs['Actual_Detection_Rate'] * 100, range(len(apple_recs)),
              color='darkblue', s=100, marker='o', label='Actual', zorder=3)
   ax4.set_yticks(range(len(apple_recs)))
   ax4.set_yticklabels(apple_recs['Pesticide'], fontsize=10)
   ax4.set_xlabel('Detection Rate (%)', fontsize=11, fontweight='bold')
   ax4.set_title('Example: Apple (AP) Recommendations\nTop 10 Pesticides',
                 fontsize=13, fontweight='bold')
   ax4.legend()
   ax4.grid(axis='x', alpha=0.3)


# Plot 5: Efficiency Gains
ax5 = fig.add_subplot(gs[1, 2])
efficiency_values = [float(s['Efficiency_Gain'].strip('%')) for s in all_savings.values()]
efficiency_sorted = sorted(zip(coverage_commodities, efficiency_values), key=lambda x: x[1], reverse=True)
comm_sorted, eff_sorted = zip(*efficiency_sorted)


colors_eff = ['darkgreen' if e > 80 else 'green' if e > 60 else 'orange' for e in eff_sorted]
ax5.barh(range(len(eff_sorted)), eff_sorted, color=colors_eff, edgecolor='black', alpha=0.7)
ax5.set_yticks(range(len(comm_sorted)))
ax5.set_yticklabels(comm_sorted, fontsize=10)
ax5.set_xlabel('Efficiency Gain (%)', fontsize=11, fontweight='bold')
ax5.set_title('Testing Efficiency Improvement\n(% reduction in tests)',
             fontsize=13, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)


for i, val in enumerate(eff_sorted):
   ax5.text(val + 1, i, f'{val:.0f}%', va='center', fontsize=9, fontweight='bold')


# Plot 6: Matrix Factorization Latent Factors
ax6 = fig.add_subplot(gs[2, 0])
factor_variance = np.var(H, axis=1)
ax6.bar(range(len(factor_variance)), factor_variance, color='purple', edgecolor='black', alpha=0.7)
ax6.set_xlabel('Latent Factor', fontsize=11, fontweight='bold')
ax6.set_ylabel('Variance', fontsize=11, fontweight='bold')
ax6.set_title('Latent Factor Analysis\n(Pesticide Patterns)', fontsize=13, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)


# Plot 7: Recommendation Accuracy
ax7 = fig.add_subplot(gs[2, 1])


mae_scores = []
for commodity in top_10_comm:
   actual = detection_matrix_filtered.loc[commodity]
   predicted = predicted_detection_df.loc[commodity]


   mae = np.mean(np.abs(actual - predicted))
   mae_scores.append(mae)


ax7.bar(range(len(mae_scores)), mae_scores, color='teal', edgecolor='black', alpha=0.7)
ax7.set_xticks(range(len(top_10_comm)))
ax7.set_xticklabels(top_10_comm, rotation=45, ha='right', fontsize=9)
ax7.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
ax7.set_title('Prediction Accuracy by Commodity\n(Lower = Better)',
             fontsize=13, fontweight='bold')
ax7.grid(axis='y', alpha=0.3)


ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')


total_samples = len(merged['sample_pk'].unique())
avg_tests_saved = np.mean([s['Tests_Saved'] for s in all_savings.values()])
total_savings = avg_tests_saved * 50 * total_samples


summary_text = f"""
COST OPTIMIZATION SUMMARY


Current Testing Strategy:
 • Average pesticides tested: {int(np.mean([s['Total_Pesticides_Tested'] for s in all_savings.values()]))}
 • Total samples: {total_samples:,}


Optimized Strategy (Top 10):
 • Average pesticides needed: 10
 • Average detection coverage: {np.mean(coverage_values):.1f}%
 • Average tests saved: {avg_tests_saved:.1f} per sample


Cost Impact:
 • Tests saved per sample: {avg_tests_saved:.1f}
 • Cost per test: $50
 • Savings per sample: ${avg_tests_saved * 50:,.0f}
 Annual Savings Potential:
 • Total savings: ${total_savings:,.0f}
 • Efficiency gain: {np.mean(efficiency_values):.1f}%


Recommendation:
 Focus testing on top 10 pesticides
 per commodity to maintain 85%+
 detection coverage while reducing
 costs by 70-80%.
"""


ax8.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
       verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


plt.savefig('recommendation_system.png', dpi=300, bbox_inches='tight')


# 7. SAVE RECOMMENDATIONS


for commodity, recs in all_recommendations.items():
   recs.to_csv(f'recommendations_{commodity}.csv', index=False)


# summary saved
summary_df = pd.DataFrame(all_savings).T
summary_df.to_csv('cost_savings_summary.csv')


# 8. INTERACTIVE LOOKUP


print("\nKey Findings:")
print(f"  • Average tests saved per sample: {avg_tests_saved:.1f}")
print(f"  • Average detection coverage: {np.mean(coverage_values):.1f}%")
print(f"  • Potential annual savings: ${total_savings:,.0f}")
print(f"  • Efficiency improvement: {np.mean(efficiency_values):.1f}%")


# Demo funct
def demo_recommendation():
   print("\n" + "="*80)
   print("DEMO: Get Recommendations for Any Commodity")
   print("="*80)
  
   if 'AP' in all_recommendations:
       print("\nExample: APPLES (AP)")
       print("-" * 40)
       recs = all_recommendations['AP']
       print(recs.to_string(index=False))
      
       savings = all_savings['AP']
       for key, val in savings.items():
           print(f"  • {key.replace('_', ' ')}: {val}")


demo_recommendation()






R code for precipitation & temperature affects 

big_data <- readr::read_csv(paste0(getwd(), "/generated.csv"))

attach(big_data)

reg <- glm(I(concen > 0) ~ Apr.Temp_2015 + May.Temp_2015 + Jun.Temp_2015 + Jul.Temp_2015 + Aug.Temp_2015 + Sep.Temp_2015 + Oct.Temp_2015 + Apr.Precip_2015 + May.Precip_2015 + Jun.Precip_2015 + Jul.Precip_2015 + Aug.Precip_2015 + Sep.Precip_2015 + Oct.Precip_2015 + state, family = binomial(link = "logit"))

detach(big_data)

summary(reg)

