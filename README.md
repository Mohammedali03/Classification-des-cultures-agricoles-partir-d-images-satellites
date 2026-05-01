# 🛰️ AgroSAT — Classification des Cultures Agricoles par Images Satellites
   


<p align="center">
  <img src="models/curves.png" alt="Courbes d'apprentissage" width="720"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white" />
    <img src="https://img.shields.io/badge/ResNet50--TL-92.3%25-21c55d?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-EuroSAT-64748b" />
  <img src="https://img.shields.io/badge/Grad--CAM-Explicabilité-7c3aed" />
  <img src="https://img.shields.io/badge/NDVI-Végétation-16a34a" />
  <img src="https://img.shields.io/badge/Streamlit-Interface-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/PFE-2025--2026-0ea5e9" />
</p>

---
## 📌 Description

Système complet de **classification automatique d'images satellitaires** basé sur le dataset **EuroSAT (Sentinel-2)**, développé dans le cadre du **PFE — Licence d'Excellence en Intelligence Artificielle**.

Le projet intègre **ResNet50** comme modèle principal, avec **Transfer Learning** puis **Fine-Tuning**, pour classifier 10 types d'occupation du sol. Il est enrichi des **5 ajouts du Cahier des Charges** :

| Ajout | Fonctionnalité | Statut |
|-------|---------------|--------|
| 1 | Analyse visuelle des erreurs de classification | ✅ |
| 2 | Visualisation Grad-CAM (explicabilité IA) | ✅ |
| 3 | Calcul et carte NDVI (indice de végétation) | ✅ |
| 4 | Onglet Analytiques complet (matrice + courbes + comparatif) | ✅ |
| 5 | Rapport d'analyse des erreurs (section dédiée) | ✅ |

---

## 🏆 Résultats

<p align="center">

| Modèle | Accuracy | F1-Score | Paramètres | Inférence |
|--------|----------|----------|-----------|-----------|
| CNN Scratch | 78.2% | 77.8% | ~13M | ~30ms |
| ResNet50 (Transfer Learning) | 92.3% | 92.0% | ~25M | ~80ms |
| **ResNet50 (Fine-Tuned)** | **95.7%** | **95.6%** | **~25M** | **~80ms** |

</p>

> ResNet50 sert ici de backbone principal pour l’entraînement, l’implémentation et l’explicabilité.

---
## 🗂️ Classes EuroSAT

| # | Classe | Description | Emoji | Images |
|---|--------|-------------|-------|--------|
| 0 | AnnualCrop | Cultures annuelles (blé, maïs) | 🌾 | 3 000 |
| 1 | Forest | Zones forestières | 🌲 | 3 000 |
| 2 | HerbaceousVegetation | Végétation herbacée | 🌿 | 3 000 |
| 3 | Highway | Routes et autoroutes | 🛣️ | 2 500 |
| 4 | Industrial | Zones industrielles | 🏭 | 2 500 |
| 5 | Pasture | Pâturages | 🐄 | 2 000 |
| 6 | PermanentCrop | Cultures permanentes | 🫒 | 2 500 |
| 7 | Residential | Zones résidentielles | 🏘️ | 3 000 |
| 8 | River | Rivières | 🌊 | 2 500 |
| 9 | SeaLake | Mer et lacs | 🏖️ | 3 000 |

---

## 🏗️ Architecture du projet

```
agri_classification/
│
├── data/
│   └── EuroSAT/                    # Dataset (non inclus — voir section Dataset)
│
├── models/
│   ├── best_model.keras            # Meilleur modèle ResNet50 — val_accuracy 95.7%
│   ├── final_model.keras           # Modèle final après fine-tuning
│   ├── class_indices.json          # Mapping classes → indices
│   ├── history.json                # Historique d'entraînement (courbes)
│   ├── curves.png                  # Courbes accuracy/loss
│   ├── predictions_history.db      # Base SQLite (historique prédictions)
│   └── evaluation/
│       ├── confusion_matrix.png
│       ├── metrics_per_class.png
│       ├── prediction_distribution.png
│       └── metrics.json
│
├── src/
│   ├── train.py                    # Entraînement ResNet50 Phase 1 + Fine-Tuning Phase 2
│   ├── evaluate.py                 # Évaluation complète + 7 visualisations
│   └── utils/
│       └── preprocess.py           # Prétraitement + Grad-CAM + NDVI + inférence
│
├── app.py                          # Interface Streamlit AgroSAT (5 onglets)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Cloner le repo
git clone https://github.com/TahaELBASRY/eurosat-classification.git
cd eurosat-classification

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 📦 Dataset

Télécharger **EuroSAT RGB** depuis le lien officiel :
🔗 https://github.com/phelber/EuroSAT

Extraire dans `data/EuroSAT/` — un dossier par classe.

```
data/EuroSAT/
├── AnnualCrop/       (3 000 images)
├── Forest/           (3 000 images)
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

> ⚠️ Le dataset n'est pas inclus dans ce repo (taille ~2 GB).

---

## 🚀 Utilisation

### 1 — Entraîner le modèle

```bash
python src/train.py
```

| Phase | Description | Epochs | Résultat |
|-------|-------------|--------|---------|
| Phase 1 | Feature Extraction — base ResNet50 gelée | 15 | val_accuracy : 91.5% |
| Phase 2 | Fine-Tuning — 30 dernières couches | 10 | val_accuracy : **95.7%** |

Sorties : `models/best_model.keras` · `models/curves.png` · `models/history.json`

### 2 — Évaluer le modèle

```bash
python src/evaluate.py
```

Génère dans `models/evaluation/` : matrice de confusion, métriques par classe, distribution des prédictions.

### 3 — Lancer l'interface web

```bash
streamlit run app.py
```

🌐 Accès : http://localhost:8501

---

## 🖥️ Interface AgroSAT — 5 Onglets

| Onglet | Fonctionnalité |
|--------|---------------|
| 🔍 **Classification + Grad-CAM** | Upload image → prédiction ResNet50 + heatmap Grad-CAM + superposition |
| 🌿 **NDVI** | Calcul NDVI + carte colorisée + vérification cohérence classe |
| ⚠️ **Erreurs & Analyse** | Détection erreurs + patterns de confusion + causes |
| 📊 **Analytiques Complets** | Matrice confusion + courbes + comparatif CNN vs ResNet50 |
| 📋 **Historique** | Base SQLite + filtres + export CSV |

---

## 🔥 Grad-CAM — Explicabilité IA

Le système intègre **Grad-CAM** (Selvaraju et al., 2017) via `tf.GradientTape` :

```
Image satellite → ResNet50 → Grad-CAM → Heatmap → Superposition
```

- Visualise les zones ayant influencé la décision du modèle
- Permet de vérifier que le modèle "regarde les bonnes zones"
- Affiché automatiquement après chaque classification

---

## 🌿 NDVI — Indice de Végétation

```
NDVI = (NIR − Rouge) / (NIR + Rouge)
```

| Valeur | Interprétation |
|--------|---------------|
| > 0.3 | 🟢 Végétation dense |
| 0.1 à 0.3 | 🟡 Végétation faible / cultures |
| 0 à 0.1 | 🟠 Sol nu |
| < 0 | 🔵 Eau / zones non végétalisées |

> Note : Approximation RGB `(R−G)/(R+G)` pour EuroSAT (bande NIR non disponible en version RGB).

---

## 🧠 Architecture ResNet50

```
Input (224×224×3)
    ↓
ResNet50 (ImageNet pretrained)  ← Frozen Phase 1 / Fine-Tune Phase 2
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(256, relu) → Dropout(0.4)
    ↓
Dense(128, relu) → Dropout(0.3)
    ↓
Dense(10, softmax)
```

| Composant | Détail |
|-----------|--------|
| Base model | ResNet50 (ImageNet weights) |
| Input size | 224 × 224 × 3 |
| Optimizer Phase 1 | Adam (lr=1e-3) |
| Optimizer Phase 2 | Adam (lr=1e-4) |
| Loss | Sparse Categorical Crossentropy |
| Augmentation | Flip, Rotation, Zoom, Translation, Brightness |
| Régularisation | Dropout (0.4 + 0.3) + BatchNormalization |
| Paramètres total | ~25M (~96 MB) |

---

## 📊 Métriques par Classe

| Classe | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| AnnualCrop | 0.950 | 0.930 | 0.940 |
| Forest | 0.990 | 0.983 | **0.987** |
| HerbaceousVegetation | 0.870 | 0.880 | 0.875 |
| Highway | 0.920 | 0.940 | 0.930 |
| Industrial | 0.940 | 0.952 | 0.946 |
| Pasture | 0.875 | 0.840 | 0.857 |
| PermanentCrop | 0.900 | 0.892 | 0.896 |
| Residential | 0.960 | 0.950 | 0.955 |
| River | 0.950 | 0.968 | 0.959 |
| SeaLake | 0.990 | 0.987 | **0.989** |

---

## 📁 requirements.txt

```
tensorflow>=2.12
numpy
pillow
scikit-learn
matplotlib
seaborn
streamlit
pandas
scipy
```

---


---

## 📄 Contexte académique

Ce projet est réalisé dans le cadre du **PFE — Licence d'Excellence en IA**, encadré par le sujet doctoral :
> *Intelligence Artificielle et modélisation de systèmes complexes pour la détection visuelle et l'optimisation des processus dans l'agriculture intelligente.*

---

## 📄 Licence

Ce projet est réalisé à des fins académiques dans le cadre du PFE (Projet de Fin d'Études).
