```
███╗   ███╗  █████╗  ██████╗██╗  ██╗██╗███╗   ██╗███████╗    ██╗     ███████╗ █████╗ ██████╗ ███╗   ██╗██╗███╗   ██╗ ██████╗ 
████╗ ████║ ██╔══██╗██╔════╝██║  ██║██║████╗  ██║██╔════╝    ██║     ██╔════╝██╔══██╗██╔══██╗████╗  ██║██║████╗  ██║██╔════╝ 
██╔████╔██║ ███████║██║     ███████║██║██╔██╗ ██║█████╗      ██║     █████╗  ███████║██████╔╝██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
██║╚██╔╝██║ ██╔══██║██║     ██╔══██║██║██║╚██╗██║██╔══╝      ██║     ██╔══╝  ██╔══██║██╔══██╗██║╚██╗██║██║██║╚██╗██║██║   ██║
██║ ╚═╝ ██║ ██║  ██║╚██████╗██║  ██║██║██║ ╚████║███████╗    ███████╗███████╗██║  ██║██║  ██║██║ ╚████║██║██║ ╚████║╚██████╔╝
╚═╝     ╚═╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 
```

# 🩺 Analyse du Dataset de Dermatologie

Analyse exploratoire et réduction de dimensionnalité sur le dataset de dermatologie de Kaggle.

## 📋 Description

Ce projet effectue une analyse complète du dataset de dermatologie incluant :

- **Chargement et exploration** des données
- **Preprocessing et nettoyage** (valeurs manquantes, outliers, normalisation)
- **Réduction de dimensionnalité** avec PCA (Analyse en Composantes Principales)
- **Visualisations** des résultats

## 🗂️ Structure du Projet

```
ML/
├── dermatology_analysis.ipynb   # Notebook principal d'analyse
├── README.md                    # Readme
├── objectifs.md                 # Objectifs
└── .gitignore                   # Fichiers ignorés par Git
```

## 🔧 Installation

### Prérequis

- Python 3.8+
- pip

### Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\Activate.ps1  # Windows PowerShell
```

### Installer les dépendances

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

## 🚀 Utilisation

1. Ouvrir le notebook `dermatology_analysis.ipynb` dans Jupyter ou VS Code
2. Exécuter les cellules dans l'ordre

## 📊 Contenu de l'Analyse

### 1. Exploration des Données
- Statistiques descriptives
- Distribution des classes
- Analyse des valeurs manquantes

### 2. Preprocessing
- Imputation des valeurs manquantes (médiane)
- Détection des outliers (méthode IQR)
- Normalisation (StandardScaler)

### 3. Analyse de Corrélation
- Matrice de corrélation
- Identification des features fortement corrélées

### 4. PCA (Analyse en Composantes Principales)
- Variance expliquée
- Visualisation 2D et 3D
- Loadings des composantes principales
- Biplot

## 📦 Bibliothèques Utilisées

| Bibliothèque | Utilisation |
|--------------|-------------|
| pandas | Manipulation de données |
| numpy | Calculs numériques |
| matplotlib | Visualisation |
| seaborn | Visualisation statistique |
| scikit-learn | PCA, preprocessing |
| kagglehub | Téléchargement du dataset |

## 📈 Résultats

L'analyse PCA permet de :
- Réduire la dimensionnalité tout en préservant 95% de la variance
- Visualiser la séparation des classes de maladies dermatologiques
- Identifier les features les plus importantes

## 📝 License

Ce projet est à des fins éducatives.

## 👤 Auteur

Projet d'analyse de données en Machine Learning.
