"""
Mental Health in Tech - Unsupervised Learning Case Study
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load the csv
df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

# rename columns to Q0, Q1... for sanity
short_names = {col: f"Q{i}" for i, col in enumerate(df.columns)}
col_mapping = {f"Q{i}": col for i, col in enumerate(df.columns)}
df.columns = [short_names[c] for c in df.columns]

print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Total missing: {df.isnull().sum().sum()}")


# --- EDA ---

# fix unrealistic ages and fill with median
age_col = 'Q55'
print(f"\nAge range before fix: {df[age_col].min()} to {df[age_col].max()}")
df.loc[(df[age_col] < 18) | (df[age_col] > 80), age_col] = np.nan
df[age_col].fillna(df[age_col].median(), inplace=True)
print(f"Age range after fix: {df[age_col].min():.0f} to {df[age_col].max():.0f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df[age_col], bins=30, edgecolor='black', alpha=0.7, color='#5DADE2')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')
axes[0].set_title('Age Distribution of Respondents')
axes[0].axvline(df[age_col].median(), color='red', linestyle='--', label=f'Median: {df[age_col].median():.0f}')
axes[0].legend()

# standardize gender field
gender_col = 'Q56'
def clean_gender(val):
    if pd.isnull(val):
        return 'Other/Prefer not to say'
    val = str(val).strip().lower()
    if val in ['male', 'm', 'man', 'male (cis)', 'cis male', 'cis man', 'mail', 'maile', 'malr']:
        return 'Male'
    elif val in ['female', 'f', 'woman', 'female (cis)', 'cis female', 'cis woman', 'femail', 'femake']:
        return 'Female'
    else:
        return 'Other/Non-binary'

df[gender_col] = df[gender_col].apply(clean_gender)
gender_counts = df[gender_col].value_counts()
axes[1].bar(gender_counts.index, gender_counts.values, color=['#5DADE2', '#F1948A', '#82E0AA'], edgecolor='black')
axes[1].set_ylabel('Count')
axes[1].set_title('Gender Distribution')
for i, v in enumerate(gender_counts.values):
    axes[1].text(i, v + 10, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_demographics.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nGender:\n{gender_counts}")

# mental health prevalence pie charts
mh_col = 'Q47'
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors_mh = ['#E74C3C', '#2ECC71', '#F39C12']

mh_counts = df[mh_col].value_counts()
axes[0].pie(mh_counts.values, labels=mh_counts.index, autopct='%1.1f%%', colors=colors_mh, startangle=90)
axes[0].set_title('Current Mental Health Disorder')

past_mh = df['Q46'].value_counts()
axes[1].pie(past_mh.values, labels=past_mh.index, autopct='%1.1f%%', colors=colors_mh, startangle=90)
axes[1].set_title('Past Mental Health Disorder')

fam_mh = df['Q45'].value_counts()
axes[2].pie(fam_mh.values, labels=fam_mh.index, autopct='%1.1f%%', colors=['#2ECC71', '#E74C3C', '#F39C12'], startangle=90)
axes[2].set_title('Family History of Mental Illness')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_mh_prevalence.png', dpi=150, bbox_inches='tight')
plt.close()

# company size + remote work
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
company_size_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
size_data = df['Q1'].dropna()
size_counts = size_data.value_counts().reindex(company_size_order)
axes[0].bar(range(len(size_counts)), size_counts.values, color='#5DADE2', edgecolor='black')
axes[0].set_xticks(range(len(size_counts)))
axes[0].set_xticklabels(company_size_order, rotation=30, ha='right')
axes[0].set_ylabel('Count')
axes[0].set_title('Company Size Distribution')

remote_counts = df['Q62'].value_counts()
axes[1].bar(remote_counts.index, remote_counts.values, color=['#F39C12', '#2ECC71', '#E74C3C'], edgecolor='black')
axes[1].set_ylabel('Count')
axes[1].set_title('Remote Work Distribution')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_workplace.png', dpi=150, bbox_inches='tight')
plt.close()

# missing values overview
fig, ax = plt.subplots(figsize=(14, 6))
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
missing_significant = missing_pct[missing_pct > 0]
ax.barh(range(len(missing_significant)), missing_significant.values, color='#E74C3C', alpha=0.7)
ax.set_yticks(range(len(missing_significant)))
ax.set_yticklabels(missing_significant.index, fontsize=7)
ax.set_xlabel('Missing Values (%)')
ax.set_title('Missing Values per Feature')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_missing_values.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFeatures with >50% missing: {(missing_pct > 50).sum()}")


# --- PREPROCESSING ---

# drop cols with >60% missing, free-text, and sparse US state columns
high_missing = missing_pct[missing_pct > 60].index.tolist()
free_text_cols = ['Q37', 'Q39', 'Q48', 'Q49', 'Q51']  # open-ended / condition descriptions
geo_sparse = ['Q58', 'Q60']  # US state (too many NaNs)
cols_to_drop = list(set(high_missing + free_text_cols + geo_sparse))

df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
print(f"\nDropped {len(cols_to_drop)} cols -> {df_clean.shape[1]} remaining")

# encode categoricals with LabelEncoder, fill missing
label_encoders = {}
df_encoded = df_clean.copy()

for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'string':
        df_encoded[col] = df_encoded[col].fillna('Unknown')
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    elif pd.api.types.is_numeric_dtype(df_encoded[col]):
        df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
    else:
        df_encoded[col] = df_encoded[col].fillna('Unknown')
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

print(f"Encoded shape: {df_encoded.shape}, missing left: {df_encoded.isnull().sum().sum()}")

# z-score standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)


# --- DIMENSIONALITY REDUCTION ---

# PCA - check how many components we need
pca_full = PCA()
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(range(1, len(cumvar)+1), cumvar, 'b-o', markersize=3)
axes[0].axhline(y=0.80, color='r', linestyle='--', label='80% variance')
axes[0].axhline(y=0.90, color='g', linestyle='--', label='90% variance')
n_80 = np.argmax(cumvar >= 0.80) + 1
n_90 = np.argmax(cumvar >= 0.90) + 1
axes[0].axvline(x=n_80, color='r', linestyle=':', alpha=0.5)
axes[0].axvline(x=n_90, color='g', linestyle=':', alpha=0.5)
axes[0].set_xlabel('Number of Components')
axes[0].set_ylabel('Cumulative Explained Variance')
axes[0].set_title('PCA - Cumulative Explained Variance')
axes[0].legend()
axes[0].set_xlim(0, len(cumvar))

axes[1].bar(range(1, 16), pca_full.explained_variance_ratio_[:15], color='#5DADE2', edgecolor='black')
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Explained Variance Ratio')
axes[1].set_title('PCA - Variance per Component (Top 15)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_pca_variance.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nPCA: {n_80} components for 80%, {n_90} for 90%")

# keep enough for ~80% variance
n_components = n_80
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
print(f"Using {n_components} components ({cumvar[n_components-1]*100:.1f}% var)")

# t-SNE on the PCA output for 2D viz
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, learning_rate='auto')
X_tsne = tsne.fit_transform(X_pca)
print("Done.")

X_pca_2d = X_pca[:, :2]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.3, s=10, c='#5DADE2')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].set_title('PCA - 2D Projection')

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.3, s=10, c='#E74C3C')
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].set_title('t-SNE - 2D Projection')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_dim_reduction_2d.png', dpi=150, bbox_inches='tight')
plt.close()


# --- CLUSTERING ---

# try K from 2 to 10, track inertia + silhouette
K_range = range(2, 11)
inertias = []
silhouettes = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_pca, labels)
    silhouettes.append(sil)
    print(f"  K={k}: Inertia={km.inertia_:.0f}, Silhouette={sil:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(K_range), inertias, 'b-o', markersize=6)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (Within-cluster SSE)')
axes[0].set_title('Elbow Method')
axes[0].set_xticks(list(K_range))

axes[1].plot(list(K_range), silhouettes, 'r-o', markersize=6)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')
axes[1].set_xticks(list(K_range))

best_k = list(K_range)[np.argmax(silhouettes)]
axes[1].axvline(x=best_k, color='green', linestyle='--', label=f'Best K={best_k}')
axes[1].legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.close()

# go with K=4 (see report for rationale)
chosen_k = 4
print(f"\nFinal K={chosen_k}")
final_km = KMeans(n_clusters=chosen_k, random_state=42, n_init=20, max_iter=500)
cluster_labels = final_km.fit_predict(X_pca)
df_clean['Cluster'] = cluster_labels
df['Cluster'] = cluster_labels

for c in range(chosen_k):
    n = (cluster_labels == c).sum()
    print(f"  Cluster {c}: {n} ({n/len(cluster_labels)*100:.1f}%)")

final_sil = silhouette_score(X_pca, cluster_labels)
print(f"Silhouette: {final_sil:.4f}")

# silhouette plot
fig, ax = plt.subplots(figsize=(8, 6))
sample_sils = silhouette_samples(X_pca, cluster_labels)
y_lower = 10
colors = plt.cm.Set2(np.linspace(0, 1, chosen_k))

for i in range(chosen_k):
    ith_sils = sample_sils[cluster_labels == i]
    ith_sils.sort()
    size_cluster_i = ith_sils.shape[0]
    y_upper = y_lower + size_cluster_i
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sils, facecolor=colors[i], alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}', fontweight='bold')
    y_lower = y_upper + 10

ax.axvline(x=final_sil, color='red', linestyle='--', label=f'Mean: {final_sil:.3f}')
ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Cluster')
ax.set_title('Silhouette Plot for K-Means Clustering')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_silhouette_plot.png', dpi=150, bbox_inches='tight')
plt.close()


# --- CLUSTER VISUALIZATION ---

# clusters on t-SNE and PCA
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
scatter_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

for c in range(chosen_k):
    mask = cluster_labels == c
    axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], alpha=0.5, s=15,
                   c=scatter_colors[c], label=f'Cluster {c} (n={mask.sum()})')
    axes[1].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], alpha=0.5, s=15,
                   c=scatter_colors[c], label=f'Cluster {c}')

axes[0].set_xlabel('t-SNE Dimension 1')
axes[0].set_ylabel('t-SNE Dimension 2')
axes[0].set_title('Clusters Visualized via t-SNE')
axes[0].legend(fontsize=8)

axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].set_title('Clusters Visualized via PCA')
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_clusters_visualization.png', dpi=150, bbox_inches='tight')
plt.close()


# --- CLUSTER PROFILING ---

orig_cols = {f"Q{i}": col for i, col in enumerate(
    pd.read_csv('mental-heath-in-tech-2016_20161114.csv', nrows=0).columns)}

print("\n--- Cluster profiles ---")
for cluster_id in range(chosen_k):
    mask = df['Cluster'] == cluster_id
    cd = df[mask]
    n = mask.sum()
    print(f"\nCluster {cluster_id} (n={n}, {n/len(df)*100:.1f}%)")
    print(f"  Age: mean={cd['Q55'].mean():.1f}, median={cd['Q55'].median():.0f}")
    print(f"  Gender: {cd['Q56'].value_counts().to_dict()}")
    print(f"  Self-employed: {(cd['Q0']==1).sum()}/{n} ({(cd['Q0']==1).sum()/n*100:.0f}%)")
    if cd['Q1'].notna().sum() > 0:
        print(f"  Company size: {cd['Q1'].dropna().value_counts().head(2).to_dict()}")
    print(f"  Current MH: {cd['Q47'].value_counts().to_dict()}")
    print(f"  Past MH: {cd['Q46'].value_counts().to_dict()}")
    print(f"  Family history: {cd['Q45'].value_counts().to_dict()}")
    print(f"  Sought treatment: {cd['Q52'].value_counts().to_dict()}")
    if cd['Q4'].notna().sum() > 0:
        print(f"  MH benefits: {cd['Q4'].dropna().value_counts().to_dict()}")
    if cd['Q6'].notna().sum() > 0:
        print(f"  Employer discusses MH: {cd['Q6'].dropna().value_counts().to_dict()}")
    if cd['Q14'].notna().sum() > 0:
        print(f"  Takes MH seriously: {cd['Q14'].dropna().value_counts().to_dict()}")
    print(f"  MH hurts career: {cd['Q40'].value_counts().to_dict()}")
    print(f"  Remote: {cd['Q62'].value_counts().to_dict()}")

# profile comparison charts
key_compare = ['Q47', 'Q46', 'Q45', 'Q50', 'Q52', 'Q40', 'Q62', 'Q56']
key_labels = ['Current MH', 'Past MH', 'Family History', 'Diagnosed', 'Sought Treatment',
              'MH Hurts Career', 'Remote Work', 'Gender']

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for idx, (col, label) in enumerate(zip(key_compare, key_labels)):
    if idx >= len(axes):
        break
    ct = pd.crosstab(df['Cluster'], df[col], normalize='index') * 100
    ct.plot(kind='bar', ax=axes[idx], legend=True, width=0.8)
    axes[idx].set_title(label, fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('Cluster')
    axes[idx].set_ylabel('%')
    axes[idx].legend(fontsize=6, loc='upper right')
    axes[idx].tick_params(axis='x', rotation=0)

plt.suptitle('Cluster Profiles - Key Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/10_cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.close()

# age by cluster
fig, ax = plt.subplots(figsize=(10, 5))
for c in range(chosen_k):
    mask = df['Cluster'] == c
    ax.hist(df.loc[mask, 'Q55'], bins=25, alpha=0.5, label=f'Cluster {c}', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.set_title('Age Distribution per Cluster')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/11_age_per_cluster.png', dpi=150, bbox_inches='tight')
plt.close()

# workplace culture bars
culture_cols = ['Q4', 'Q6', 'Q7', 'Q10', 'Q14']
culture_labels = ['MH Benefits', 'Discusses MH', 'MH Resources', 'Negative Consequences', 'Takes MH Seriously']

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for idx, (col, label) in enumerate(zip(culture_cols, culture_labels)):
    data_sub = df[df[col].notna()]
    ct = pd.crosstab(data_sub['Cluster'], data_sub[col], normalize='index') * 100
    ct.plot(kind='bar', ax=axes[idx], legend=True, width=0.8)
    axes[idx].set_title(label, fontsize=9, fontweight='bold')
    axes[idx].set_xlabel('Cluster')
    axes[idx].set_ylabel('%')
    axes[idx].legend(fontsize=5, loc='upper right')
    axes[idx].tick_params(axis='x', rotation=0)

plt.suptitle('Workplace Mental Health Culture by Cluster', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/12_workplace_culture.png', dpi=150, bbox_inches='tight')
plt.close()

# quick summary table
print("\n--- Summary ---")
summary_data = []
for c in range(chosen_k):
    mask = df['Cluster'] == c
    cd = df[mask]
    summary_data.append({
        'Cluster': c,
        'Size': mask.sum(),
        'Pct': f"{mask.sum()/len(df)*100:.1f}%",
        'Mean Age': f"{cd['Q55'].mean():.1f}",
        'Current MH Yes%': f"{(cd['Q47']=='Yes').sum()/mask.sum()*100:.0f}%",
        'Sought Treatment%': f"{(cd['Q52']==1).sum()/mask.sum()*100:.0f}%",
        'Self-employed%': f"{(cd['Q0']==1).sum()/mask.sum()*100:.0f}%",
        'Male%': f"{(cd['Q56']=='Male').sum()/mask.sum()*100:.0f}%"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
print(f"\nSilhouette: {final_sil:.4f}")
print(f"PCA components: {n_components} ({cumvar[n_components-1]*100:.1f}% var)")
summary_df.to_csv(f'{OUTPUT_DIR}/cluster_summary.csv', index=False)

print(f"\nAll figures in {OUTPUT_DIR}/")
