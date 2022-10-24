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

Notre jeu de données est une collection d'images de feuilles de vigne, réparties en deux classes : les feuilles malsaines provenant de plantes affectées par la maladie d'Esca et les feuilles saines. 
Les données, fournies par STMicroelectronics et disponibles sur l'Internet, ont été collectées pour être utilisées dans le cadre d'un projet de recherche développé conjointement par le département d'ingénierie de l'information de l'université polytechnique des Marches, à Ancône (Italie), et STMicroelectronics, Italie, avec la coopération de l'entreprise vinicole Umani Ronchi SPA, Osimo, Ancône, Marches, Italie [y].
<div>
 <img src="https://user-images.githubusercontent.com/29697453/197035673-efdc06f0-42c9-43a9-9291-4e17d4d43c57.jpg" width="505"/>
 <br>Figure 1 - Exemple de feuillage sain 
</div>

<div>
 <img src="https://user-images.githubusercontent.com/29697453/197035607-89dcb607-d79b-40ce-8d77-697dffe0c397.jpg" width="505"/>
 <br>Figure 2 - Exemple de feuillage malade
</div>

## Pré-requis de logiciel 

<ol>
  <li>MXCube ST</li>
  <li>STM32 Cube IDE</li>
  <li>Environnement Python 3.8</li>
</ol>

## Pré-requis matériels 
<ol>
  <li>Nucléo F411RE Board</li>
</ol>


## Modèle de machine learning 
Les phases de machine learning en Python sont très importantes dans ce projet, néanmoins pour faciliter la compréhension de notre modèle par le lecteurs nous avons ci-dessous une figure du déroulement des étapes de notre CNN qui sera intégré, testé et validé.
Les opérations effectuées sont des opérations classiques de ML étudiées lors de notre deuxième année d'études. 
Dans l'entrée du modèle, nous pouvons voir dans la première ligne conv2d, où nous avons un OutPut Shape qui doit être respectée, nous avons donc dû normaliser les données de l'ensemble de données avant de l'utiliser, en se rappelant que nous ne pouvons pas changer ces caractéristiques intrinsèques du modèle concerné. 
De plus, sur la sortie (Activation), nous avons un OutPutShape (None, 2), ce qui signifie que nous avons un résultat avec deux classes, parce que pour résoudre notre problème, étant donné une image, nous voulons avoir la réponse malade et non malade et les probabilités respectives pour chaque décision.
<br><br>
<div>
 <img src="https://user-images.githubusercontent.com/29697453/197405758-045a8e8a-c821-4d51-a5d9-d16e4c4b7078.png" width="505"/>
 <br>Figure 3 - Phases du modèle ML
</div>

## Méthodologie 

Dans la figure ci-dessous, nous pouvons voir un schéma du déroulement de notre méthodologie de travail. Nous commençons par la phase de Data Augmentation, après ce prétraitement sur le jeu de données, nous avons la phase d'entraînement du modèle sur notre ordinateur personnel. Ces deux phases se déroulent exclusivement dans l'environnement Python. Dans la troisième étape, la configuration dans CubeMX, nous commençons à penser au niveau du matériel, grâce à cet outil de STMicroelectronics et au paquet Cube-MX-AI pour notre card.
Chaque phase de notre méthodologie sera expliquée en détail par la suite.

![312665544_1818175281854117_2918214121483767220_n](https://user-images.githubusercontent.com/29697453/197516827-8c6c291a-ec17-46c7-9427-0d1be58a6694.png)
<br>Figure 4 - Flux de la méthodologie utilisée 

### Data augmentation

Pour pouvoir entraîner notre modèle, il nous faut un dataset avec une bonne quantité et surtout une bonne qualité des données. En effet, il est préférable d'appliquer les transformations pour augmenter la diversité et donc le champ d'apprentissage de notre modèle. Nous pouvons alors améliorer efficacement le processus d'apprentissage puisqu'il en résulte en plus d'échantillons d'entrainement pour le modèle.


<div>
 <img src="https://user-images.githubusercontent.com/29697453/197527116-3cc51d18-d08c-4299-b9b8-0ad71e191481.png" width="500"/>
 <br>Figure 5 - Possibilité de transformations sur les images du jeu de données
</div>

<br>
<br>

La base de code DataAugmentation a été fournie par l'enseignant pendant le cours et nous avons apporté les modifications nécessaires pour appliquer les changements à notre jeu de données. Dans l'image ci-dessus, vous pouvez voir la liste des opérations disponibles. Ces opérations vont générer de nouvelles images, dans notre cas, nous les appliquons de manière aléatoire horizontalFlips, rotations, shearRange, zooms et nous améliorons la luminosité, le contraste et la saturation des images.

### Entraînement du modèle 

![312583538_5612427252207688_858929209927743802_n](https://user-images.githubusercontent.com/29697453/197641363-95820b8d-0b46-4ac1-9f6c-ad9b8fd58a38.png)
<br>Figure 6 - La précision du modèle 

### Configurations sur CubeMX


### Communication 
![312585686_3307257036203296_3800052736440173189_n (1)](https://user-images.githubusercontent.com/29697453/197642471-3ddad507-1b9c-4d6b-879a-9eddf08bc989.png)
<br>Figure 7 - Les valeurs d'une communication UART réussie

### Interprétation des résultats 

<div>
 <img src="https://user-images.githubusercontent.com/29697453/197642840-4f3e77d3-024a-4f63-944b-6da9e36a621c.png" width="600"/>
 <br>Figure 8 - La précision du modèle où les prévisions sont correctes 
</div>

<br><br>

<div>
 <img src="https://user-images.githubusercontent.com/29697453/197644334-6761962b-2e75-4a2c-bb65-b43e8fa2a3c5.png" width="600"/>
 <br>Figure 9 - La précision du modèle où les prévisions sont pas correctes
</div>

## Attaques sur le jeu de données

La méthode du signal de gradient rapide (FGSM) est une méthode simple mais efficace pour générer des images contradictoires. 
FGSM exploite les gradients d'un réseau neuronal pour construire une image adverse.

Essentiellement, le FGSM calcule les gradients d'une fonction de perte par rapport à l'image d'entrée, puis utilise le signal des gradients pour créer une nouvelle image, c'est-à-dire l'image adverse, qui maximise la perte.

Le résultat est une image de sortie qui, selon l'oeil humain, semble identique à l'original, mais qui amène le réseau neuronal à faire une prédiction incorrecte.

![Capture d’écran 2022-10-25 004706](https://user-images.githubusercontent.com/29697453/197644180-d9d77457-e1a8-4b68-8552-e9c10cdf263e.png)
<br>Figure 10 - Génération d'erreurs à injecter dans l'image de jeu de données

![Capture d’écran 2022-10-25 004554](https://user-images.githubusercontent.com/29697453/197643997-3505da8b-8c1b-4a49-aac3-41595bac6b81.png)
<br>Figure 11 - Génération d'images en faisant varier les valeurs d'epsilon et en augmentant l'intensité de l'erreur 

## Conclusion

Nous avons eu l'occasion de découvrir l'IA embarquée, un sujet en pleine croissance sur le marché technologique et qui intéresse plusieurs entreprises. La croissance des objets connectés et des systèmes de plus en plus intelligents rend le sujet intéressant et stimulant à la fois, avec de nombreuses possibilités de création et d'innovation. En même temps, nous pouvons voir comment la technologie peut aider les problèmes humains, dans ce cas, dans l'agriculture et la plantation. En tant qu'ingénieurs, nous devons toujours aller de l'avant en réfléchissant à la manière d'aider les gens et notre environnement, sans avoir besoin d'utiliser des produits toxiques et en étant même capables de prédire les problèmes avant même qu'ils ne se produisent, dans ce contexte, l'AI Embarquée se révèle être un outil formidable.
<br>
Nous pourrions entraîner un modèle, l'attaquer, étudier les moyens d'attaque et nous pourrions travailler avec une carte destinée à l'IA embarquée. 


## Références 
[1] TinyML Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers - disponible sur https://tinymlbook.com/wp-content/uploads/2020/11/TinyML_preview.pdf 
<br>
[2] Inaugural tinyML 2019 - https://www.tinyml.org/event/summit-2019
<br>
[3] TinyML Summit - https://www.tinyml.org/
<br>
[4] ESCA-dataset - disponible sur https://data.mendeley.com/datasets/89cnxc58kj/1
<br>


