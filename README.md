# Prédiction du Cancer de la Prostate — ML vs Deep Learning

Comparaison rigoureuse de plusieurs approches de Machine Learning et Deep Learning pour la classification du diagnostic (bénin/malin) à partir de biomarqueurs cliniques.

##  Description

Ce projet explore quelle famille de modèles est la plus adaptée à un problème médical sur un **petit jeu de données** (100 patients, 8 variables numériques). L'objectif n'est pas seulement de maximiser l'accuracy, mais de prioriser le **rappel (recall)**, car en contexte médical un faux négatif (cas malin non détecté) est l'erreur la plus coûteuse.

##  Dataset

- 100 observations, 9 variables descriptives (radius, texture, perimeter, area, smoothness, compactness, symmetry, fractal_dimension) + variable cible `diagnosis_result` (B/M)
- Données cliniques tabulaires, sans valeurs manquantes

##  Méthodologie

1. **Exploration et nettoyage** des données (vérification des types, des distributions, des corrélations)
2. **Mise à l'échelle (scaling)** des variables pour les modèles sensibles aux échelles
3. **Entraînement et comparaison** de 5 approches :
   - Régression Logistique
   - K-Nearest Neighbors (KNN)
   - LightGBM
   - XGBoost (avec tuning d'hyperparamètres)
   - Réseau de neurones (MLP) en PyTorch
4. **Évaluation** sur un jeu de test indépendant via accuracy, recall, precision et matrice de confusion

##  Résultats

| Modèle | Accuracy (Test) | Recall (Classe 1) | Observation |
|---|---|---|---|
| **XGBoost (tuné)** | **87 %** | **95,4 %** | Meilleur modèle, généralisation optimale |
| KNN (K=12) | 86,7 % | 91,0 % | Très solide sur petit dataset tabulaire |
| Régression Logistique | 83,3 % | 91,0 % | Interprétable, légèrement moins précise sur la classe minoritaire |
| LightGBM | 76,7 % | 81,8 % | Overfitting marqué (CV 90 % vs Test 77 %) |
| MLP (PyTorch) | 73,3 % | 81,8 % | Dataset trop restreint pour exploiter le Deep Learning |

##  Enseignements clés

- **La règle du "small data"** : sur un dataset de cette taille (~150 échantillons), les modèles complexes (Deep Learning, LightGBM) sont sujets à l'overfitting et sont surpassés par des approches plus simples et mieux régularisées.
- **Priorité métier au recall** : en détection de cancer, manquer un cas malin (faux négatif) est plus grave qu'une fausse alerte. XGBoost tuné minimise ces erreurs avec un recall de 95,4 %.
- **Importance des variables** : selon XGBoost, le **perimeter** et la **compactness** de la tumeur sont les prédicteurs les plus déterminants du diagnostic.

##  Modèle retenu

**XGBoost (tuné)** est sélectionné comme modèle final : il offre le meilleur compromis entre sécurité diagnostique (recall élevé) et précision globale.

##  Pistes d'amélioration

- Collecter davantage d'échantillons pour la classe minoritaire afin de stabiliser encore la précision
- Explorer des techniques de rééquilibrage de classes (SMOTE, pondération de la loss)

##  Stack technique

Python · Pandas · Scikit-Learn · XGBoost · LightGBM · PyTorch · Matplotlib / Seaborn

##  Structure

```
.
├── prostate-cancer-ml-vs-dl.ipynb   # Notebook complet (EDA, modélisation, évaluation)
├── README.md
└── requirements.txt
```

## ▶️ Utilisation

```bash
pip install -r requirements.txt
jupyter notebook prostate-cancer-ml-vs-dl.ipynb
```
