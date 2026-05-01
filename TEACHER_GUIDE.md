# AgroSAT — Guide de présentation pour le professeur

## 1) Vue d’ensemble du projet

AgroSAT est un projet de **classification d’images satellitaires** basé sur le dataset **EuroSAT**.  
L’objectif est de reconnaître automatiquement le type d’occupation du sol à partir d’une image satellite RGB.

Le projet classe **10 catégories** :

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

Le cœur du système repose sur **ResNet50 en Transfer Learning**, avec une phase de **feature extraction** suivie d’une phase de **fine-tuning**.

---

## 2) Pourquoi ce projet est intéressant

Ce projet montre plusieurs compétences importantes :

- utilisation d’un **dataset satellite réel**
- entraînement d’un modèle de **deep learning**
- passage d’un modèle CNN simple vers un modèle plus performant
- mise en place d’une **explicabilité** avec Grad-CAM
- ajout d’une **analyse NDVI** pour relier la prédiction à une logique végétale
- création d’une **interface web Streamlit** pour la démonstration
- sauvegarde des prédictions et historique des analyses

En résumé, ce n’est pas juste un modèle de classification.  
C’est un petit système complet d’**analyse d’images satellites**.

---

## 3) Dataset utilisé : EuroSAT

Le dataset utilisé est **EuroSAT RGB**, basé sur des images Sentinel-2.

### Structure générale

Chaque classe contient plusieurs milliers d’images.  
Le dataset est organisé par dossiers, un dossier par classe.

### Pourquoi EuroSAT

- dataset adapté à la télédétection
- classes bien connues et équilibrées
- bon cas d’usage pour tester le transfert d’apprentissage
- utile pour des applications agricoles, urbaines et environnementales

### Ce que représente une image

Une image EuroSAT est une petite image satellite RGB qui contient des indices visuels sur :

- la végétation
- l’urbanisation
- les routes
- les zones d’eau
- les terrains agricoles
- les surfaces industrielles

---

## 4) Modèle choisi : ResNet50

Le modèle principal est **ResNet50**, utilisé en **Transfer Learning**.

### Pourquoi ResNet50

ResNet50 est un excellent choix parce que :

- il est puissant pour la reconnaissance visuelle
- il apprend des représentations profondes et stables
- il évite en partie le problème de disparition du gradient grâce aux connexions résiduelles
- il fonctionne bien sur des tâches de classification d’images
- il est suffisamment robuste pour le fine-tuning sur EuroSAT

### Idée du Transfer Learning

Au lieu d’entraîner un réseau entièrement depuis zéro :

1. on part d’un modèle déjà entraîné sur ImageNet
2. on remplace la tête de classification
3. on adapte le modèle à EuroSAT

Cela permet :

- de gagner du temps
- d’obtenir de meilleures performances
- de réduire le besoin en données
- de commencer avec des caractéristiques visuelles déjà apprises

---

## 5) Architecture du réseau

L’architecture finale est :

```text
Input 224×224×3
→ ResNet50 (ImageNet pretrained)
→ GlobalAveragePooling2D
→ BatchNormalization
→ Dense(256, relu)
→ Dropout(0.4)
→ Dense(128, relu)
→ Dropout(0.3)
→ Dense(10, softmax)
```

### Rôle des couches

- **ResNet50** : extrait les caractéristiques visuelles principales
- **GlobalAveragePooling2D** : réduit la dimension des cartes de caractéristiques
- **BatchNormalization** : stabilise l’apprentissage
- **Dense(256)** et **Dense(128)** : apprennent la classification spécifique au dataset
- **Dropout** : limite le surapprentissage
- **Softmax final** : donne la probabilité pour chaque classe

---

## 6) Étapes d’entraînement

Le training est fait en **deux phases**.

### Phase 1 — Feature Extraction

Dans cette phase :

- la base ResNet50 est **gelée**
- seules les couches finales sont entraînées
- le but est d’apprendre la nouvelle tête de classification

#### Pourquoi

Cette étape permet de partir d’un modèle stable et de l’adapter au dataset EuroSAT sans casser les poids préentraînés.

#### Paramètres principaux

- **Epochs** : 15
- **Optimizer** : Adam
- **Learning rate** : `1e-3`
- **Loss** : Sparse Categorical Crossentropy
- **Metric** : Accuracy

#### Ce que l’on observe

Le modèle apprend rapidement les grandes distinctions entre classes :

- végétation vs zones urbaines
- eau vs terre
- routes vs zones agricoles

---

### Phase 2 — Fine-Tuning

Dans cette phase :

- on dégele la base ResNet50
- on garde gelées les couches de début
- on entraîne seulement les dernières couches du backbone

Dans le code, les **40 dernières couches** sont prises en compte, avec les BatchNormalization conservées gelées.

#### Pourquoi

Le fine-tuning permet d’adapter plus finement le modèle aux motifs spécifiques d’EuroSAT.

#### Paramètres principaux

- **Epochs** : 10
- **Optimizer** : Adam
- **Learning rate** : `1e-4`
- **Loss** : Sparse Categorical Crossentropy
- **Metric** : Accuracy

#### Pourquoi un learning rate plus faible

Parce qu’en fine-tuning on ne veut pas modifier brutalement les poids préentraînés.  
Un petit learning rate permet d’affiner le modèle sans le déstabiliser.

---

## 7) Prétraitement des images

Le prétraitement est important pour que l’entrée soit compatible avec ResNet50.

### Pipeline d’inférence

1. conversion en RGB
2. redimensionnement en `224×224`
3. conversion en tableau numpy
4. application de `preprocess_input`
5. ajout de la dimension batch

### Point important

Le projet insiste sur un point essentiel :

> `preprocess_input` doit être appliqué **une seule fois**.

Si on l’applique deux fois, les images deviennent mal normalisées et les prédictions se dégradent fortement.

### Prétraitement pendant l’entraînement

- augmentation de données sur le train set
- normalisation ResNet50
- pas d’augmentation sur validation

### Augmentation utilisée

- flip horizontal
- rotation
- zoom
- translation
- brightness

#### Pourquoi

L’augmentation améliore la généralisation et rend le modèle plus robuste aux variations d’images satellites.

---

## 8) Stratégie de validation

Le dataset est séparé en :

- **80% entraînement**
- **20% validation**

La validation sert à vérifier si le modèle généralise bien sur des images non vues pendant l’entraînement.

---

## 9) Métriques utilisées

Les métriques suivies sont :

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

### Interprétation simple

- **Accuracy** : proportion globale de bonnes prédictions
- **Precision** : quand le modèle prédit une classe, à quel point il a raison
- **Recall** : combien d’éléments réels d’une classe il retrouve
- **F1-score** : compromis entre Precision et Recall

---

## 10) Résultats obtenus

Selon les valeurs présentes dans le projet, le modèle ResNet50 atteint environ :

- **Accuracy** : 92.3% à 95.7% selon la phase / meilleure sauvegarde
- **F1-score** : autour de 92%

### Comment expliquer cela au professeur

Tu peux dire :

> Le modèle a d’abord appris la tâche grâce au transfer learning, puis il a été affiné avec le fine-tuning. C’est cette deuxième étape qui permet de monter en performance.

### Pourquoi les résultats peuvent paraître très élevés

- le transfer learning part d’un très bon point de départ
- le dataset EuroSAT est assez adapté à ce type de classification
- les classes sont visuellement distinctes dans plusieurs cas
- le modèle a été régularisé avec dropout et batch normalization

---

## 11) Pourquoi les résultats sont crédibles

Si un professeur demande pourquoi l’accuracy est élevée, la bonne réponse est :

- le modèle est basé sur une architecture très solide
- il profite d’un entraînement en deux étapes
- les images satellitaires contiennent des structures visuelles assez cohérentes
- le modèle a été évalué avec des métriques complémentaires
- la matrice de confusion permet de vérifier les erreurs réelles

Il ne faut pas dire seulement “le modèle est bon”.  
Il faut dire **pourquoi** il est bon.

---

## 12) Problèmes de confusion entre classes

Certaines classes sont naturellement plus difficiles à distinguer.

### Exemples

- **HerbaceousVegetation vs Pasture**
- **AnnualCrop vs PermanentCrop**
- **Highway vs River** parfois selon la forme

### Pourquoi

Ces classes peuvent partager :

- des couleurs proches
- des textures similaires
- des formes géométriques proches
- une faible résolution spatiale

### Ce que cela montre

Le modèle n’est pas parfait, mais ses erreurs sont cohérentes avec la nature visuelle du dataset.

---

## 13) Explicabilité avec Grad-CAM

Le projet inclut **Grad-CAM**.

### Rôle

Grad-CAM permet de voir **où le modèle regarde** pour prendre sa décision.

### Utilité

- vérifier que le modèle se base sur les bonnes zones
- rassurer sur l’interprétabilité
- mieux comprendre une prédiction incorrecte

### Chaîne de fonctionnement

```text
Image → ResNet50 → gradients → heatmap → superposition
```

### Ce que tu peux dire au professeur

> Grad-CAM montre les régions les plus influentes pour la prédiction. Cela aide à vérifier que le modèle ne prédit pas au hasard et qu’il s’appuie sur des zones pertinentes de l’image.

---

## 14) Analyse NDVI

Le projet ajoute aussi un calcul **NDVI**.

### Formule

```text
NDVI = (NIR - Rouge) / (NIR + Rouge)
```

### Dans EuroSAT RGB

Comme les données RGB ne contiennent pas la vraie bande NIR, le projet utilise une **approximation visuelle** :

```text
(R - G) / (R + G)
```

Ce n’est pas un NDVI scientifique exact, mais c’est une **approximation utile pour l’analyse visuelle**.

### À quoi ça sert

- vérifier la cohérence d’une classe végétale
- comparer la prédiction avec un indicateur de végétation
- repérer des incohérences éventuelles

### Interprétation simplifiée

- NDVI élevé → végétation dense
- NDVI moyen → cultures / végétation faible
- NDVI faible → sol nu
- NDVI négatif → eau ou zone non végétalisée

---

## 15) Interface Streamlit

L’interface permet de :

- téléverser une image
- obtenir une prédiction
- voir la confiance du modèle
- afficher le top des probabilités
- générer une carte Grad-CAM
- calculer un NDVI
- consulter l’historique des prédictions
- visualiser les métriques et les courbes d’apprentissage

### Pourquoi une interface est importante

Parce qu’elle transforme le modèle en outil démontrable.  
Le professeur peut voir le résultat en direct, ce qui rend le projet plus concret.

---

## 16) Script d’entraînement : résumé simple à dire à l’oral

Tu peux résumer le training comme ça :

> J’ai commencé par charger le dataset EuroSAT, puis j’ai séparé les données en train et validation. J’ai utilisé ResNet50 préentraîné sur ImageNet, j’ai d’abord gelé la base pour apprendre la nouvelle tête de classification, puis j’ai dégelé une partie du backbone pour faire du fine-tuning. J’ai utilisé l’augmentation de données sur le train set uniquement, et j’ai surveillé l’accuracy, la loss et les métriques par classe.

---

## 17) Ce que font les fichiers principaux

### `src/train.py`

- charge EuroSAT
- applique augmentation + preprocessing
- entraîne ResNet50 en deux phases
- sauvegarde le meilleur modèle
- sauvegarde les courbes d’apprentissage
- enregistre les classes

### `src/evaluate.py`

- charge le meilleur modèle
- évalue sur le dataset de validation
- calcule accuracy, precision, recall, F1
- génère la matrice de confusion
- exporte les graphiques et métriques JSON

### `utils/preprocess.py`

- charge le modèle
- prétraite une image pour l’inférence
- fait la prédiction
- gère le calcul d’évaluation si nécessaire

### `app.py`

- interface Streamlit
- classification interactive
- Grad-CAM
- NDVI
- historique des prédictions

---

## 18) Ce qu’il faut dire si le professeur demande : “Pourquoi ResNet50 ?”

Réponse courte :

> J’ai choisi ResNet50 parce qu’elle est robuste, efficace et très adaptée au transfer learning. Les connexions résiduelles facilitent l’apprentissage, et son extraction de caractéristiques fonctionne bien sur les images satellites.

---

## 19) Ce qu’il faut dire si le professeur demande : “Pourquoi fine-tuning ?”

Réponse courte :

> Le fine-tuning permet d’adapter le modèle aux motifs spécifiques d’EuroSAT. La première phase apprend la tête de classification, puis la deuxième affine les couches hautes pour améliorer la performance finale.

---

## 20) Ce qu’il faut dire si le professeur demande : “Pourquoi vos résultats sont élevés ?”

Réponse courte :

> Les résultats sont bons parce que ResNet50 est déjà très performant, que le dataset EuroSAT est bien structuré, et que le modèle a été entraîné en deux étapes avec augmentation, régularisation et validation rigoureuse.

---

## 21) Ce qu’il faut dire si le professeur demande : “Est-ce que le modèle est parfait ?”

Réponse courte :

> Non, comme tout modèle, il a des limites. Certaines classes très proches visuellement peuvent être confondues, surtout en RGB. C’est pour cela que j’ai ajouté Grad-CAM et l’analyse NDVI pour mieux comprendre les décisions.

---

## 22) Conclusion

Ce projet montre un pipeline complet de deep learning appliqué à la télédétection :

- préparation des données
- entraînement en deux phases
- évaluation rigoureuse
- interprétabilité du modèle
- interface de démonstration

C’est un projet solide parce qu’il ne se limite pas à un simple score d’accuracy.  
Il explique aussi **comment** le modèle apprend et **pourquoi** il prend ses décisions.

---

## 23) Mini script de présentation orale

Tu peux l’utiliser presque tel quel :

> Mon projet consiste à classer des images satellites EuroSAT en 10 catégories. J’ai utilisé ResNet50 en transfer learning. J’ai d’abord gelé le backbone pour entraîner la tête de classification, puis j’ai fait un fine-tuning sur les dernières couches. J’ai appliqué l’augmentation de données uniquement sur le train set et j’ai utilisé un preprocessing cohérent avec ResNet50. Ensuite, j’ai évalué le modèle avec accuracy, precision, recall et F1-score, et j’ai ajouté Grad-CAM pour l’explicabilité ainsi qu’un calcul NDVI pour vérifier la cohérence végétale.

---

## 24) Fiche ultra courte pour réviser

- **Dataset** : EuroSAT RGB
- **Classes** : 10
- **Modèle** : ResNet50
- **Méthode** : Transfer Learning + Fine-Tuning
- **Entrée** : image 224×224×3
- **Prétraitement** : `preprocess_input`
- **Phase 1** : backbone gelé
- **Phase 2** : dégel partiel
- **Métriques** : Accuracy, Precision, Recall, F1
- **Explicabilité** : Grad-CAM
- **Indice végétation** : NDVI approximatif
- **Interface** : Streamlit

---

## 25) Points forts à mettre en avant

- projet complet et bien structuré
- bon choix de modèle
- logique scientifique claire
- résultats élevés et cohérents
- interface moderne et démonstrable
- ajout d’explicabilité et d’analyse métier

---

## 26) Limites à mentionner si on te questionne

- NDVI exact impossible en RGB sans vraie bande NIR
- certaines classes restent proches visuellement
- les résultats dépendent de la qualité des images
- le dataset reste limité par rapport à des données multi-spectrales complètes

---

## 27) Message final simple

> Ce projet est un système complet de classification satellitaire basé sur ResNet50, entraîné en deux phases, évalué avec des métriques solides, expliqué avec Grad-CAM et enrichi avec une analyse NDVI pour donner du sens aux prédictions.
