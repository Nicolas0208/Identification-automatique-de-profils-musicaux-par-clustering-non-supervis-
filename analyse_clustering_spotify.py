import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (silhouette_score, davies_bouldin_score, 
                            calinski_harabasz_score, classification_report)
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

# SECTION 1 : CHARGEMENT ET PRÉPARATION
# 

CHEMIN_FICHIER = r'C:\Users\nicol\Documents\Drive\Cours\Master 2\ML\genres_v2.csv'

print("ÉTAPE 1 : CHARGEMENT DES DONNÉES")

def charger_csv_robuste(chemin):
    """Charge le CSV avec format européen (virgule décimale)"""
    
    separateurs = [';', ',', '\t']
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    
    print(f"\n→ Recherche de la configuration optimale...")
    
    for encoding in encodings:
        for sep in separateurs:
            for decimal in [',', '.']:
                try:
                    # Test
                    df_test = pd.read_csv(chemin, 
                                         encoding=encoding, 
                                         sep=sep,
                                         decimal=decimal,
                                         on_bad_lines='skip',
                                         nrows=5)
                    
                    # Vérifier qu'on a les bonnes colonnes
                    if 'danceability' in df_test.columns and df_test.shape[1] > 10:
                        # Charger tout
                        df = pd.read_csv(chemin, 
                                       encoding=encoding, 
                                       sep=sep,
                                       decimal=decimal,
                                       on_bad_lines='skip')
                        return df
                        
                except Exception:
                    continue
    
    raise ValueError("Impossible de charger le CSV")

# Charger
df = charger_csv_robuste("C:\ML\genres_v2.csv")
print(f"\n Dataset chargé : {df.shape[0]:,} morceaux × {df.shape[1]} colonnes")


# Colonnes
print(f"\nColonnes disponibles :")
for i, col in enumerate(df.columns[:10], 1):
    print(f"  {i}. {col}")
if len(df.columns) > 10:
    print(f"  ... et {len(df.columns)-10} autres colonnes")

audio_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
]

# Vérifier
missing = [col for col in audio_features if col not in df.columns]
if missing:
    print(f"\n Colonnes manquantes : {missing}")
    exit(1)

print(f"\n Toutes les colonnes audio sont présentes")

# Extraire
X = df[audio_features].copy()
y_genre = df['genre'].copy()

print(f"Features extraites : {X.shape}")

# CONVERSION FORCÉE (crucial pour format européen)
print("\n Conversion des colonnes en format numérique")

conversion_needed = False

for col in audio_features:
    if X[col].dtype == 'object':
        if not conversion_needed:
            print("\n  Colonnes nécessitant une conversion :")
            conversion_needed = True
        
        print(f"    • {col}")
        
        # Conversion : remplacer , par . puis convertir
        X[col] = X[col].astype(str).str.replace(',', '.').str.strip()
        X[col] = pd.to_numeric(X[col], errors='coerce')

if not conversion_needed:
    print(" Toutes les colonnes sont déjà numériques")
else:
    print("\n Conversions effectuées")

# Nettoyer les NaN
nan_total = X.isnull().sum().sum()
if nan_total > 0:
    print(f"\n {nan_total} valeurs NaN détectées")
    
    # Afficher par colonne
    nan_per_col = X.isnull().sum()
    for col in nan_per_col[nan_per_col > 0].index:
        print(f"  {col} : {nan_per_col[col]} NaN")
    
    # Option 1 : Supprimer les lignes
    rows_before = len(X)
    X = X.dropna()
    y_genre = y_genre.loc[X.index]
    rows_after = len(X)
    
    print(f"\n  → {rows_before - rows_after} lignes supprimées")
    print(f"  Dataset nettoyé : {rows_after:,} morceaux")

# Vérification finale
print("\n Vérification finale des types :")
all_numeric = True
for col in audio_features:
    dtype = X[col].dtype
    is_numeric = np.issubdtype(dtype, np.number)
    status = "✓" if is_numeric else "✗"
    print(f"  {status} {col:20s} : {dtype}")
    if not is_numeric:
        all_numeric = False

if not all_numeric:
    print("\n ERREUR : Certaines colonnes ne sont pas numériques")
    print("Vérifiez manuellement le fichier CSV")
    exit(1)

print(f"\n Toutes les colonnes sont numériques")
print(f" Dataset final : {len(X):,} morceaux × {len(audio_features)} features")

# STANDARDISATION
print("\n Standardisation des données")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=audio_features)

print(f"Données standardisées : {X_scaled.shape}")
print(f" Moyenne ≈ 0, Écart-type ≈ 1 pour toutes les features")

# SECTION 2 : DÉTECTION ET TRAITEMENT DES OUTLIERS
print("\n ÉTAPE 2 : DÉTECTION DES OUTLIERS")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso_forest.fit_predict(X_scaled)

n_outliers = (outlier_labels == -1).sum()
n_inliers = (outlier_labels == 1).sum()

print(f"\n Outliers détectés : {n_outliers:,} ({n_outliers/len(X)*100:.2f}%)")
print(f" Inliers conservés : {n_inliers:,} ({n_inliers/len(X)*100:.2f}%)")

# Créer versions avec/sans outliers
X_scaled_no_outliers = X_scaled[outlier_labels == 1]
X_no_outliers = X[outlier_labels == 1]
y_genre_no_outliers = y_genre[outlier_labels == 1]

print(f"\n DÉCISION : On va comparer les résultats AVEC et SANS outliers")
print(f"   Version 1 : {len(X):,} morceaux (avec outliers)")
print(f"   Version 2 : {len(X_no_outliers):,} morceaux (sans outliers)")

# SECTION 3 : COMPARAISON DE 4 ALGORITHMES DE CLUSTERING
print("\n ÉTAPE 3 : COMPARAISON D'ALGORITHMES (k=5)")

# On teste sur la version SANS outliers pour de meilleurs résultats
X_test = X_scaled_no_outliers
optimal_k = 5

results_algorithms = {}

# ALGORITHME 1 : K-Means
print("\n Test K-Means...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_test)

sil_kmeans = silhouette_score(X_test, labels_kmeans)
db_kmeans = davies_bouldin_score(X_test, labels_kmeans)
ch_kmeans = calinski_harabasz_score(X_test, labels_kmeans)

results_algorithms['K-Means'] = {
    'labels': labels_kmeans,
    'silhouette': sil_kmeans,
    'davies_bouldin': db_kmeans,
    'calinski_harabasz': ch_kmeans,
    'n_clusters': optimal_k
}

print(f"  Silhouette       : {sil_kmeans:.4f}")
print(f"  Davies-Bouldin   : {db_kmeans:.4f}")
print(f"  Calinski-Harabasz: {ch_kmeans:.0f}")

# ALGORITHME 2 : Gaussian Mixture Model (GMM)
print("\n Test Gaussian Mixture Model")
gmm = GaussianMixture(n_components=optimal_k, random_state=42, covariance_type='full')
labels_gmm = gmm.fit_predict(X_test)

sil_gmm = silhouette_score(X_test, labels_gmm)
db_gmm = davies_bouldin_score(X_test, labels_gmm)
ch_gmm = calinski_harabasz_score(X_test, labels_gmm)

results_algorithms['GMM'] = {
    'labels': labels_gmm,
    'silhouette': sil_gmm,
    'davies_bouldin': db_gmm,
    'calinski_harabasz': ch_gmm,
    'n_clusters': optimal_k
}

print(f"  Silhouette       : {sil_gmm:.4f}")
print(f"  Davies-Bouldin   : {db_gmm:.4f}")
print(f"  Calinski-Harabasz: {ch_gmm:.0f}")

# ALGORITHME 3 : DBSCAN
print("\n Test DBSCAN...")

# Trouver eps optimal avec k-distance graph
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=10)
neighbors_fit = neighbors.fit(X_test)
distances, indices = neighbors_fit.kneighbors(X_test)
distances = np.sort(distances[:, -1], axis=0)
eps_optimal = np.percentile(distances, 90)  # Heuristique

dbscan = DBSCAN(eps=eps_optimal, min_samples=50)
labels_dbscan = dbscan.fit_predict(X_test)

n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_dbscan = list(labels_dbscan).count(-1)

print(f"  Paramètres : eps={eps_optimal:.3f}, min_samples=50")
print(f"  Clusters détectés : {n_clusters_dbscan}")
print(f"  Points de bruit   : {n_noise_dbscan:,}")

if n_clusters_dbscan > 1:
    # Calculer métriques (sans les points de bruit)
    mask_no_noise = labels_dbscan != -1
    X_test_no_noise = X_test[mask_no_noise]
    labels_dbscan_no_noise = labels_dbscan[mask_no_noise]
    
    sil_dbscan = silhouette_score(X_test_no_noise, labels_dbscan_no_noise)
    db_dbscan = davies_bouldin_score(X_test_no_noise, labels_dbscan_no_noise)
    ch_dbscan = calinski_harabasz_score(X_test_no_noise, labels_dbscan_no_noise)
    
    print(f"  Silhouette       : {sil_dbscan:.4f}")
    print(f"  Davies-Bouldin   : {db_dbscan:.4f}")
    print(f"  Calinski-Harabasz: {ch_dbscan:.0f}")
    
    results_algorithms['DBSCAN'] = {
        'labels': labels_dbscan,
        'silhouette': sil_dbscan,
        'davies_bouldin': db_dbscan,
        'calinski_harabasz': ch_dbscan,
        'n_clusters': n_clusters_dbscan,
        'noise_points': n_noise_dbscan
    }
else:
    print(" DBSCAN n'a pas trouvé de structure significative")

# ALGORITHME 4 : Agglomerative Clustering (Hiérarchique)
print("\n Test Agglomerative Clustering (Hiérarchique)...")
agg = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
labels_agg = agg.fit_predict(X_test)

sil_agg = silhouette_score(X_test, labels_agg)
db_agg = davies_bouldin_score(X_test, labels_agg)
ch_agg = calinski_harabasz_score(X_test, labels_agg)

results_algorithms['Hierarchical'] = {
    'labels': labels_agg,
    'silhouette': sil_agg,
    'davies_bouldin': db_agg,
    'calinski_harabasz': ch_agg,
    'n_clusters': optimal_k
}

print(f"  Silhouette       : {sil_agg:.4f}")
print(f"  Davies-Bouldin   : {db_agg:.4f}")
print(f"  Calinski-Harabasz: {ch_agg:.0f}")

# Comparaison et choix du meilleur
print("\n COMPARAISON DES ALGORITHMES")

comparison_df = pd.DataFrame({
    'Algorithme': list(results_algorithms.keys()),
    'Silhouette': [results_algorithms[k]['silhouette'] for k in results_algorithms.keys()],
    'Davies-Bouldin': [results_algorithms[k]['davies_bouldin'] for k in results_algorithms.keys()],
    'Calinski-Harabasz': [results_algorithms[k]['calinski_harabasz'] for k in results_algorithms.keys()],
    'N_Clusters': [results_algorithms[k]['n_clusters'] for k in results_algorithms.keys()]
})

print("\n" + comparison_df.to_string(index=False))

# Score composite (normaliser et combiner)
comparison_df['Score_Composite'] = (
    (comparison_df['Silhouette'] - comparison_df['Silhouette'].min()) / 
    (comparison_df['Silhouette'].max() - comparison_df['Silhouette'].min()) +
    (comparison_df['Davies-Bouldin'].max() - comparison_df['Davies-Bouldin']) / 
    (comparison_df['Davies-Bouldin'].max() - comparison_df['Davies-Bouldin'].min()) +
    (comparison_df['Calinski-Harabasz'] - comparison_df['Calinski-Harabasz'].min()) / 
    (comparison_df['Calinski-Harabasz'].max() - comparison_df['Calinski-Harabasz'].min())
) / 3

print("\n SCORE COMPOSITE (moyenne normalisée des 3 métriques) :")
for algo, score in zip(comparison_df['Algorithme'], comparison_df['Score_Composite']):
    print(f"  {algo:20s} : {score:.4f}")

best_algo = comparison_df.loc[comparison_df['Score_Composite'].idxmax(), 'Algorithme']
print(f"\n MEILLEUR ALGORITHME : {best_algo}")

# Garder les labels du meilleur algorithme
best_labels = results_algorithms[best_algo]['labels']

# SECTION 4 : DENDROGRAMME (SI HIÉRARCHIQUE GAGNE)
print("\n ÉTAPE 4 : DENDROGRAMME (Clustering Hiérarchique)")

# Prendre un échantillon pour le dendrogramme (trop lent sur 40K+)
sample_size = min(1000, len(X_test))
sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
X_sample = X_test[sample_indices]

print(f"\n Calcul du dendrogramme sur {sample_size} morceaux (échantillon)...")
linkage_matrix = linkage(X_sample, method='ward')

plt.figure(figsize=(16, 8))
dendrogram(linkage_matrix, no_labels=True)
plt.title('Dendrogramme - Clustering Hiérarchique (échantillon)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Index des morceaux', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.axhline(y=150, color='r', linestyle='--', linewidth=2, 
            label='Coupe suggérée (k=5)')
plt.legend()
plt.tight_layout()
plt.savefig('v2_dendrogramme.png', dpi=300, bbox_inches='tight')
print(" Dendrogramme sauvegardé : v2_dendrogramme.png")
plt.close()

# SECTION 5 : VISUALISATION UMAP (MEILLEURE QUE PCA)
print("\n ÉTAPE 5 : VISUALISATION UMAP")

try:
    import umap
    print("\n Calcul de la projection UMAP (peut prendre 2-3 minutes)...")
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_umap = reducer.fit_transform(X_test)
    
    print(" UMAP calculé avec succès")
    
    # Visualisation
    plt.figure(figsize=(16, 10))
    colors_map = plt.cm.tab10(range(optimal_k))
    
    for cluster_id in range(optimal_k):
        mask = best_labels == cluster_id
        if mask.sum() > 0:
            plt.scatter(embedding_umap[mask, 0], embedding_umap[mask, 1],
                       c=[colors_map[cluster_id]], 
                       label=f'Cluster {cluster_id} ({mask.sum():,})',
                       alpha=0.6, s=15)
    
    plt.xlabel('UMAP Dimension 1', fontsize=13, fontweight='bold')
    plt.ylabel('UMAP Dimension 2', fontsize=13, fontweight='bold')
    plt.title(f'Visualisation UMAP - {best_algo} (5 clusters)', 
              fontsize=17, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('v2_umap_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Visualisation UMAP sauvegardée : v2_umap_visualization.png")
    plt.close()
    
    umap_available = True
    
except ImportError:
    print("\n UMAP non installé (pip install umap-learn)")
    print("   Utilisation de PCA comme fallback...")
    umap_available = False
    
    # Fallback PCA
    pca = PCA(n_components=2, random_state=42)
    embedding_pca = pca.fit_transform(X_test)
    
    plt.figure(figsize=(14, 10))
    for cluster_id in range(optimal_k):
        mask = best_labels == cluster_id
        if mask.sum() > 0:
            plt.scatter(embedding_pca[mask, 0], embedding_pca[mask, 1],
                       label=f'Cluster {cluster_id}', alpha=0.6, s=20)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12, fontweight='bold')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12, fontweight='bold')
    plt.title(f'Visualisation PCA - {best_algo}', fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('v2_pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()


# SECTION 6 : NOMMAGE AUTOMATIQUE INTELLIGENT
print("\n ÉTAPE 6 : NOMMAGE AUTOMATIQUE INTELLIGENT DES CLUSTERS")

# Ajouter les labels au DataFrame
X_no_outliers_labeled = X_no_outliers.copy()
X_no_outliers_labeled['cluster'] = best_labels

# Profils moyens
cluster_profiles = X_no_outliers_labeled.groupby('cluster').mean()

def nommer_cluster_automatique_v2(cluster_id, profile, genres_top3):
    
    # Extraire features
    dance = profile['danceability']
    energy = profile['energy']
    acoustic = profile['acousticness']
    instrumental = profile['instrumentalness']
    valence = profile['valence']
    tempo = profile['tempo']
    loud = profile['loudness']
    speech = profile['speechiness']
    live = profile['liveness']
    duration = profile['duration_ms'] / 60000
    
    # Scoring multi-critères
    scores = {}
    
    # PROFIL 1 : RAP ÉNERGIQUE
    score_rap_energique = 0
    if speech > 0.20: score_rap_energique += 3  # Forte présence de paroles parlées
    if dance > 0.70: score_rap_energique += 2   # Très dansant
    if energy > 0.60: score_rap_energique += 2  # Énergique
    if tempo > 140: score_rap_energique += 1    # Rapide
    if valence > 0.45: score_rap_energique += 1 # Positif
    if acoustic < 0.20: score_rap_energique += 1  # Électronique
    scores['Rap Énergique & Dansant'] = score_rap_energique
    
    # PROFIL 2 : RAP SOMBRE/MÉLANCOLIQUE
    score_rap_sombre = 0
    if speech > 0.12: score_rap_sombre += 2
    if dance > 0.65: score_rap_sombre += 1
    if valence < 0.35: score_rap_sombre += 3     # Très sombre
    if energy < 0.65: score_rap_sombre += 2     # Pas trop énergique
    if acoustic < 0.20: score_rap_sombre += 1
    scores['Rap Sombre & Mélancolique'] = score_rap_sombre
    
    # PROFIL 3 : INTENSE & AGRESSIF (Metal/Hardstyle)
    score_intense = 0
    if energy > 0.85: score_intense += 3         # Très énergique
    if loud > -5: score_intense += 2             # Très fort
    if valence < 0.35: score_intense += 2        # Sombre
    if dance < 0.55: score_intense += 2          # Peu dansant
    if acoustic < 0.10: score_intense += 1       # Électronique
    scores['Intense & Agressif (Rock/Metal/Hardstyle)'] = score_intense
    
    # PROFIL 4 : HOUSE/TECHHOUSE JOYEUX
    score_house = 0
    if dance > 0.74: score_house += 3            # Très dansant
    if energy > 0.75: score_house += 2           # Énergique
    if valence > 0.55: score_house += 3          # Joyeux
    if tempo >= 120 and tempo <= 135: score_house += 2  # Tempo house
    if acoustic < 0.10: score_house += 1
    scores['Dansant & Joyeux (House/Techhouse)'] = score_house
    
    # PROFIL 5 : ÉLECTRO INSTRUMENTALE
    score_electro_instr = 0
    if instrumental > 0.65: score_electro_instr += 3  # Très instrumental
    if duration > 5: score_electro_instr += 2        # Long
    if energy > 0.75: score_electro_instr += 2       # Énergique
    if acoustic < 0.10: score_electro_instr += 1
    if speech < 0.10: score_electro_instr += 1
    scores['Électro Instrumentale Longue (Techno/Psy)'] = score_electro_instr
    
    # PROFIL 6 : ACOUSTIQUE & CALME
    score_acoustique = 0
    if acoustic > 0.40: score_acoustique += 3     # Acoustique
    if energy < 0.55: score_acoustique += 3       # Calme
    if loud < -8: score_acoustique += 2           # Doux
    if valence < 0.40: score_acoustique += 1      # Mélancolique
    scores['Calme & Acoustique (Ballades)'] = score_acoustique
    
    # PROFIL 7 : LIVE/FESTIVAL
    score_live = 0
    if live > 0.50: score_live += 4               # Très live
    if energy > 0.80: score_live += 2
    scores['Live/Festival (Trance/Psy)'] = score_live
    
    # Trouver le profil avec le meilleur score
    best_profile = max(scores, key=scores.get)
    best_score = scores[best_profile]
    confidence = min(best_score / 10, 1.0)  # Normaliser sur 0-1
    
    # Ajouter genres dominants
    genres_str = ', '.join(genres_top3)
    
    return {
        'nom': best_profile,
        'score': best_score,
        'confidence': confidence,
        'genres_dominants': genres_str
    }

# Analyser et nommer chaque cluster
print("ANALYSE ET NOMMAGE DES CLUSTERS")

df_analysis = pd.DataFrame({
    'cluster': best_labels,
    'genre': y_genre_no_outliers.values
})

cluster_names = {}

for cluster_id in range(optimal_k):
    profile = cluster_profiles.loc[cluster_id]
    
    # Top 3 genres
    cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
    top_3_genres = cluster_data['genre'].value_counts().head(3).index.tolist()
    
    # Nommer
    result = nommer_cluster_automatique_v2(cluster_id, profile, top_3_genres)
    cluster_names[cluster_id] = result
    
    print(f"CLUSTER {cluster_id}")
    print(f" NOM AUTOMATIQUE : {result['nom']}")
    print(f" Score de confiance : {result['confidence']:.2%}")
    print(f" Genres dominants : {result['genres_dominants']}")
    print(f"\n Caractéristiques audio :")
    print(f"  Danceability : {profile['danceability']:.3f}")
    print(f"  Energy       : {profile['energy']:.3f}")
    print(f"  Valence      : {profile['valence']:.3f}")
    print(f"  Acousticness : {profile['acousticness']:.3f}")
    print(f"  Instrumental : {profile['instrumentalness']:.3f}")


# SECTION 7 : VALIDATION PAR CLASSIFICATION SUPERVISÉE
print("\n ÉTAPE 7 : VALIDATION PAR CLASSIFICATION SUPERVISÉE")

# Encoder les genres
le = LabelEncoder()
y_encoded = le.fit_transform(y_genre_no_outliers)

# Split train/test
X_train, X_test_clf, y_train, y_test_clf = train_test_split(
    X_no_outliers, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Standardiser
X_train_scaled = scaler.fit_transform(X_train)
X_test_clf_scaled = scaler.transform(X_test_clf)

# Random Forest
print("\n Entraînement Random Forest pour classification de genres...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_train_scaled, y_train)

y_pred_clf = rf_clf.predict(X_test_clf_scaled)
accuracy = (y_pred_clf == y_test_clf).mean()

print(f" Accuracy : {accuracy:.2%}")

# Prédire les probabilités pour tous les morceaux
X_all_scaled_no_outliers = scaler.transform(X_no_outliers)
genre_probas = rf_clf.predict_proba(X_all_scaled_no_outliers)

# Créer DataFrame
df_proba = pd.DataFrame(genre_probas, columns=le.classes_)
df_proba['cluster'] = best_labels

# Moyennes par cluster
cluster_genre_probas = df_proba.groupby('cluster').mean()

print("PROBABILITÉS MOYENNES DE GENRE PAR CLUSTER :")
print(cluster_genre_probas.round(3))

# Heatmap
plt.figure(figsize=(16, 8))
sns.heatmap(cluster_genre_probas.T * 100, annot=True, fmt='.1f', 
            cmap='YlOrRd', cbar_kws={'label': 'Probabilité (%)'})
plt.title(f'Prédiction de genres par cluster - Validation ML ({best_algo})',
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Genre', fontsize=12, fontweight='bold')
plt.xlabel('Cluster', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('v2_validation_classification.png', dpi=300, bbox_inches='tight')
print("\n Heatmap de validation sauvegardée")
plt.close()


# SECTION 8 : GRAPHIQUE COMPARATIF FINAL
print("\n ÉTAPE 8 : GÉNÉRATION DU RAPPORT COMPARATIF")

# Graphique de comparaison des algorithmes
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Silhouette
algos = list(results_algorithms.keys())
sil_scores = [results_algorithms[k]['silhouette'] for k in algos]
axes[0].barh(algos, sil_scores, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Silhouette Score', fontsize=11, fontweight='bold')
axes[0].set_title('Silhouette Score\n(Plus élevé = mieux)', fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Davies-Bouldin
db_scores = [results_algorithms[k]['davies_bouldin'] for k in algos]
axes[1].barh(algos, db_scores, color='coral', edgecolor='black')
axes[1].set_xlabel('Davies-Bouldin Index', fontsize=11, fontweight='bold')
axes[1].set_title('Davies-Bouldin Index\n(Plus bas = mieux)', fontsize=12, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

# Calinski-Harabasz
ch_scores = [results_algorithms[k]['calinski_harabasz'] for k in algos]
axes[2].barh(algos, ch_scores, color='lightgreen', edgecolor='black')
axes[2].set_xlabel('Calinski-Harabasz Score', fontsize=11, fontweight='bold')
axes[2].set_title('Calinski-Harabasz Score\n(Plus élevé = mieux)', fontsize=12, fontweight='bold')
axes[2].grid(axis='x', alpha=0.3)

plt.suptitle('Comparaison des Algorithmes de Clustering', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('v2_algorithms_comparison.png', dpi=300, bbox_inches='tight')
print(" Graphique comparatif sauvegardé")
plt.close()

# Sauvegarder les résultats
results_summary = pd.DataFrame({
    'Cluster_ID': range(optimal_k),
    'Nom_Automatique': [cluster_names[i]['nom'] for i in range(optimal_k)],
    'Confidence': [cluster_names[i]['confidence'] for i in range(optimal_k)],
    'Genres_Dominants': [cluster_names[i]['genres_dominants'] for i in range(optimal_k)],
    'Taille': [np.sum(best_labels == i) for i in range(optimal_k)]
})

results_summary.to_csv('v2_clusters_summary.csv', index=False)
print("\n Résumé sauvegardé : v2_clusters_summary.csv")
