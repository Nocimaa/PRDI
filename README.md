# PRDI

Groupe:
- Charles Prioux
- Cyprien Deruelle
- Lucas Collemare
- Antoine Malmezac

Ce README resume les etapes de `main.ipynb` et l analyse associee pour reconstruire un CNN a la main (convolution, pooling, couches denses), entrainer sur MNIST, puis evaluer les performances.

## Etapes et analyses

1) Contexte et objectif
- But: reconstituer un CNN en recodant les briques clefs (convolution, pooling, fully connected) avec PyTorch, puis evaluer sur MNIST.
- Analyse: cela permet de comprendre le fonctionnement interne des couches (forme des tenseurs, padding, stride, backprop) tout en gardant l ecosysteme PyTorch pour l entrainement.

2) Importations
- Modules charges: torch, nn, DataLoader/random_split, datasets/transforms, matplotlib et LRScheduler du cours.
- Analyse: PyTorch sert pour les tenseurs et l autograd, torchvision pour MNIST et la normalisation, matplotlib pour les courbes de suivi.

3) Schedulers de taux d apprentissage
- `LRScheduler` et `LRSchedulerOnPlateau` implementent la logique de decay simple ou conditionnelle.
- Analyse: la version "OnPlateau" reduit le LR quand la loss stagne, ce qui stabilise l entrainement et limite les sur-apprentissages.

4) Couches et composants re-implementes
- `Conv2d`, `ReLU`, `MaxPool2d`, `Flatten`, `Linear`, `Dropout`, `CrossEntropyLoss`.
- Analyse:
  - `Conv2d` construit les patches via `unfold` et effectue un produit matriciel, ce qui explicite la convolution.
  - `MaxPool2d` applique un max sur les fenetres, avec gestion du padding par `-inf`.
  - `CrossEntropyLoss` calcule le log-softmax puis la NLL, ce qui montre la stabilite numerique via le "shift".

5) Optimiseur Adam adapte
- Classe `Adam` custom avec moments `m` et `v`.
- Analyse: adapte au dimensionnement des images et a l apprentissage du reseau perso; il remplace l Adam PyTorch standard pour garder la coherence du projet "from scratch".

6) Architecture du CNN
- `SimpleCNN` = 2 blocs Conv-ReLU-MaxPool puis un classifieur Dense.
- Analyse: les sorties 28x28 passent a 7x7 apres deux pools, d ou le flatten en 64*7*7 vers une couche dense 128, puis sortie 10 classes MNIST.

7) Fonctions d entrainement et d evaluation
- `train_one_epoch`, `evaluate`, `evaluate_metrics`, `plot_training_results`.
- Analyse:
  - Separation train/eval garantit une mesure fiable.
  - `evaluate_metrics` fournit accuracy par classe pour detecter les chiffres faibles.
  - Le plot combine loss, accuracy et learning rate pour lire la dynamique d entrainement.

8) Pipeline principal
- Device: CPU ou CUDA.
- Transformations: `ToTensor` + normalisation MNIST (moy=0.1307, std=0.3081).
- Split: 90% train, 10% validation via `random_split`.
- Entrainement: 5 epoques, suivi des courbes, scheduler "OnPlateau".
- Evaluation finale: loss/accuracy globales + accuracy par classe.
- Analyse: le suivi du LR et des courbes permet de verifier la convergence; les metriques par classe valident que le reseau generalise de facon homogene.

## Execution
- Lancer `main()` dans `main.ipynb` pour reproduire l entrainement et les figures.
