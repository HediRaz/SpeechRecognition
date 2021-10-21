# SpeechRecognition

## Tasks:
- Preprocess le dataset pour pouvoir y accéder simplement
  - Déplacer les dossiers
  - Faire une fonction qui va chercher audio + label dans le dataset
  - Convertir les label au bon format : chaine d'id des lettres

- Créer un data loader : https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
  - Load audio + label
  - Mettre au bon format : reéchantillonage à 16kHz + format stereo 2 channels
  - Resize à une taille définie
  - Audio augmentation (time shift)
  - Convertir en spectrogramme
  - SpecAugmentation
  - MFCC ?

- Implémenter le réseau de neurones
  - Une convolution avec un grand stride sur le temps
  - Reshape
  - Des convolutions
  - Couche BLSTM ou conformers
  - Prédiction
 
- Implémenter la loss CTC

- Implémenter la metric WER
