# PRIM Logbook

## ??/09/2024
- Read articles

## ??/09/2024
- Meeting

## ??/09/2024
- Read articles

## 10/10/2024
- Quick meeting

## 16/10/2024
- Kickoff meeting

## 17/10/2024
- First connexion to GPU cluster

## 18/10/2024
- Reproduce Ballé results:
  - Test compressAI manually (index error)
  - Use compressAI example notebooks (dimension error)
    - Image size multiple of 64 (https://github.com/InterDigitalInc/CompressAI/issues/252)
    - Error depends of the model...

## 19/10/2024
- Reproduce Ballé results:
  - Connect to GPU cluster
> ssh -Y fallemand-24@gpu-gw.enst.fr
  - Find INFRES storage (uid: 87287 -> ir700)
  - Connect to interactive mode
  - Install conda
  - Create balle_reproduction env
  - Clone PRIM GitHub repository
