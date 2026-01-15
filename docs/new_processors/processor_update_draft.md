# Wstepny przeglad: modyfikacje i nowe procesory

## Modyfikacje istniejacych procesorow
- [x] `encoder`: dodac encodery z docs/new_processors: leave-one-out, M-estimate, James-Stein, GLMM, WoE, binary/base-n; opcjonalnie dirty_cat (similarity/minhash) jako nowa metoda.
- [x] `rank_features`: dodac wariant `gauss_rank` (RankGauss) i/lub tryb fit na train + transform na val/test.
- [x] `scaler`: dodac power transforms (Box-Cox, Yeo-Johnson) oraz jawny preset RankGauss, jesli ma byc obok `quantile_normal`.
- [x] `outlier_handler`: rozszerzyc winsoryzacje o MAD/gaussian/percentyle; dodac tryb "trim" (drop rows) obok clip/set_na/flag_only.
- [x] `feature_group_agg` / `groupwise_normalizer`: dodac diff/ratio wzgledem mediany lub innych statystyk (min/max/median/quantile) oraz z-score z mediany/MAD.
- [x] `feature_selector`: dodac selekcje na bazie null importances i/lub permutation importance.
- [x] `sanity_check` (lub osobny prosty moduł): progowe dropna dla wierszy/kolumn (z docs: "Usuwanie brakow").

## Nowe procesory
- [x] `dae_embeddings`: Denoising Autoencoder (swap noise) generujacy embeddingi do dolaczenia do danych.
- [x] `knn_graph_features`: cechy KNN (odleglosc do k-tego sasiada, srednia odleglosc, gestosc, ewent. cechy grafowe).
- [x] `noise_injection`: augmentacja treningu (gauss/swap noise) jako osobny krok (tylko train).
- [x] `mixup_augmentation`: MixUp dla tabular (tylko train, wymaga targetu).
- [x] `pseudo_labeling`: semi-supervised (train + pseudo-labeled test); raczej etap modelu, ale do decyzji.
