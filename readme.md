
# End-to-end Speaker Verification (Triplet-loss)

This repository implements an end-to-end speaker verification system using triplet-loss metric learning on spectrogram inputs. The primary goal is to learn compact speaker embeddings such that utterances from the same speaker are close in embedding space while utterances from different speakers are far apart.

The project contains utilities for data preparation, triplet generation, model building blocks, and training examples (scripts and notebooks) to experiment with embedding models and verification evaluation. The dataset is organized by speaker folders containing raw audio and precomputed spectrograms; triplet lists are provided for training and testing.

