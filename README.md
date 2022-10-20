# **Projet IA Embarqué - EMSE**

## Objectifs
Dans ce projet, nous étudions d’intégrer une modèle CNN dans la carte STM32L4, pour le dataset dédié à la détection de maladie de la vigne par l'analyse d'images de feuilles sur le jeu de données esca_dataset.

## Jeu de données
<div>
 <img src="https://user-images.githubusercontent.com/29697453/197035673-efdc06f0-42c9-43a9-9291-4e17d4d43c57.jpg" width="505"/>
 <img src="https://user-images.githubusercontent.com/29697453/197035607-89dcb607-d79b-40ce-8d77-697dffe0c397.jpg" width="505"/>
</div>

## esca_dataset


### Augmentation

Pour pouvoir entraîner notre modèle, il nous faut un dataset avec une bonne quantité et surtout une bonne qualité des données. En effet, il est préférable d'appliquer les transformations pour augmenter la diversité et donc le champ d'apprentissage de notre modèle. Nous pouvons alors améliorer efficacement le processus d'apprentissage puisqu'il en résulte en plus d'échantillons d'entrainement pour le modèle.

