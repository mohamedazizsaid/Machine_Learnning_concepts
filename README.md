<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
</div>

```text
███╗   ███╗  █████╗  ██████╗██╗  ██╗██╗███╗   ██╗███████╗    ██╗     ███████╗ █████╗ ██████╗ ███╗   ██╗██╗███╗   ██╗ ██████╗ 
████╗ ████║ ██╔══██╗██╔════╝██║  ██║██║████╗  ██║██╔════╝    ██║     ██╔════╝██╔══██╗██╔══██╗████╗  ██║██║████╗  ██║██╔════╝ 
██╔████╔██║ ███████║██║     ███████║██║██╔██╗ ██║█████╗      ██║     █████╗  ███████║██████╔╝██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
██║╚██╔╝██║ ██╔══██║██║     ██╔══██║██║██║╚██╗██║██╔══╝      ██║     ██╔══╝  ██╔══██║██╔══██╗██║╚██╗██║██║██║╚██╗██║██║   ██║
██║ ╚═╝ ██║ ██║  ██║╚██████╗██║  ██║██║██║ ╚████║███████╗    ███████╗███████╗██║  ██║██║  ██║██║ ╚████║██║██║ ╚████║╚██████╔╝
╚═╝     ╚═╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 
```

# 🩺 Système Expert Dermatologique : Classification & Segmentation

Ce projet vise à développer un pipeline complet d'**Analyse de Données et de Machine Learning** sur le célèbre dataset de dermatologie de Kaggle. L'objectif est double : comprendre la structure sous-jacente des symptômes patients (segmentation) et concevoir un outil robuste d'aide au diagnostic clinique (classification supervisée).

---

## 📋 Tableau de Bord du Projet

| Piliers de l'analyse | Tâches accomplies | Outils & Algorithmes |
|----------------------|-------------------|----------------------|
| **1. Data Quality** | Imputation médiane, gestion outliers, standardisation. | `pandas`, `StandardScaler` |
| **2. Exploration** | Matrice de corrélation, réduction de dimension (Variance expliquée : 95%). | `seaborn`, `PCA` |
| **3. Segmentation** | Regroupement des profils patients, méthodes de densité vs centroïdes. | `K-Means`, `DBSCAN`, `Silhouette` |
| **4. Modélisation** | Entraînement avec validation croisée stratifiée (5-fold) et interprétation via matrice de confusion. | `SVM`, `RandomForest`, `XGBoost` |
| **5. Aide au diagnostic**| Déploiement simulé d'une fonction de prédiction avec top 3 diagnostics probables et filtres d'alerte. | Automatisé via Python (+ `predict_proba`) |

---

## 🗂️ Structure du Projet

```bash
ML/
├── dermatology_analysis.ipynb   # Notebook principal contenant l'intégralité du pipeline
├── objectifs.md                 # Grille des objectifs BO/DSO (Business/Data Science)
├── README.md                    # Documentation du projet
└── plots/                       # Graphiques auto-générés (PCA, Corrélations...)
```

---

## ⚙️ Déploiement Local & Installation

### Prérequis
- **Python 3.8+**

### 1. Cloner et préparer l'environnement

```bash
# Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate        # Sous Linux/Mac
.\venv\Scripts\Activate.ps1     # Sous Windows PowerShell
```

### 2. Installer les dépendances

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost kagglehub jupyter
```

### 3. Lancer l'analyse
Ouvrez le notebook pour exécuter toutes les étapes et générer les rapports :
```bash
jupyter notebook dermatology_analysis.ipynb
```

---

## 📊 Résultats & Performances

### 1. Segmentation (Non Supervisée)
- **K-Means (K=4)** : Révèle des sous-groupes structurels pertinents dans les symptômes, avec un *Silhouette Score* robuste.
- **DBSCAN** : Utile pour identifier des patients cibles formant des clusters de densité spécifique et repérer de potentiels cas atypiques (bruit).

### 2. Classification Supervisée (Modèle Champion)
Après un benchmark complet (Arbre de Décision, Random Forest, XGBoost et SVM) consolidé par une validation croisée stratifiée (5-fold) :
- **Meilleur Modèle** : **SVM (Noyau RBF)**
- **Précision globale (Accuracy)** : `~97%`
- **F1-macro** : `> 0.96` (très bonne gestion du léger déséquilibre des classes).

### 3. Impact Clinique (Système d'aide au diagnostic)
Le projet intègre une fonction d'aide à la décision :
- Fournit le **Top 3** des diagnostics avec scores de probabilité associés.
- **Alertes intelligentes** : Notification visuelle si un patient présente un cas ambigu (diagnostic majoritaire évalué à &lt; 50%).
- Extraction des **Features Importances** (Top 10 des symptômes les plus discriminants définis par Random Forest).

---

## 📝 Traçabilité Business / Data Science
Ce projet a été mené avec rigueur méthodique : toutes les métriques et résultats techniques (DSO) ont été cartographiés vers des objectifs "métier" médicaux (BO) validant ainsi la pertinence d'une IA d'assistance clinique. (Voir tableau Section 10 du notebook).

---

## 👤 Auteur
**Mohamed Aziz Said** — Projet d'analyse et de Deep Learning/Machine Learning.
