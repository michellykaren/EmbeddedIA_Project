# **Projet IA Embarqué - EMSE**
Élèves Michelly PEREIRA et Zhuan

## Introduction 
Le Machine Learning (ML) nous permet d'apprendre aux ordinateurs à faire des prédictions et à prendre des décisions sur la base de données et à tirer des enseignements de leurs expériences. Ces dernières années, des optimisations incroyables ont été apportées aux algorithmes de machine learning, aux cadres logiciels et au matériel embarqué. Grâce à cela, l'exécution de réseaux neuronaux profonds et d'autres algorithmes de machine learning complexes est possible sur des dispositifs à faible puissance comme les microcontrôleurs. En plus, il convient de noter que l'apprentissage automatique intégré est également connu sous le nom de TinyML [1]. 

Le TinyML est défini de manière générale comme un domaine en pleine expansion de technologies et d'applications d'apprentissage machine, y compris des circuits intégrés, des algorithmes et des logiciels dédiés capables d'effectuer une analyse sur dispositif des données de capteurs, par exemple de vision, d'audio, de biomédical, etc. avec une puissance extrêmement faible, ce qui permet une variété de cas d'utilisation en permanence et de cibler les dispositifs fonctionnant sur batterie. 

Un événement intéressant aura lieu en 2019 [2], le sommet inaugural tinyML avec la présence d'entreprises de renommée mondiale telles que Google, Apple, Arm, Microsoft, Qualcomm, etc. Selon [3], la communauté a montré un très fort intérêt pour le sujet, révélant : 
<br>(i) un matériel minuscule capable d'apprentissage automatique devient suffisamment bon pour de nombreuses applications commerciales et de nouvelles architectures, par exemple, l'informatique en mémoire
<br>(ii) des progrès significatifs dans les algorithmes, les réseaux et les modèles jusqu'à 100 Ko et moins 
<br>(iii) des applications initiales à faible consommation dans les domaines de la vision et de l'audio. Les progrès techniques et le développement de l'écosystème témoignent d'un élan croissant.


## Objectifs
Dans ce projet, nous étudions d’intégrer une modèle CNN dans la carte STM32L4, pour le dataset dédié à la détection de maladie de la vigne par l'analyse d'images de feuilles sur le jeu de données esca_dataset.

## Jeu de données
<div>
 <img src="https://user-images.githubusercontent.com/29697453/197035673-efdc06f0-42c9-43a9-9291-4e17d4d43c57.jpg" width="505"/>
 <br>Figure x
</div>

<div>
 <img src="https://user-images.githubusercontent.com/29697453/197035607-89dcb607-d79b-40ce-8d77-697dffe0c397.jpg" width="505"/>
 <br>Figure x
</div>

![summary](https://user-images.githubusercontent.com/29697453/197405758-045a8e8a-c821-4d51-a5d9-d16e4c4b7078.png)

## esca_dataset

## Méthodologie 




### Augmentation

Pour pouvoir entraîner notre modèle, il nous faut un dataset avec une bonne quantité et surtout une bonne qualité des données. En effet, il est préférable d'appliquer les transformations pour augmenter la diversité et donc le champ d'apprentissage de notre modèle. Nous pouvons alors améliorer efficacement le processus d'apprentissage puisqu'il en résulte en plus d'échantillons d'entrainement pour le modèle.

## Références 
[1] TinyML Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers - disponible sur https://tinymlbook.com/wp-content/uploads/2020/11/TinyML_preview.pdf 
<br>
[2] Inaugural tinyML 2019 - https://www.tinyml.org/event/summit-2019
[3] tinyML Summit - https://www.tinyml.org/


