# Solar-Image-Restoration-with-the-Cycle-GAN

## Prerequisites
- PyTorch
- torchvision

## DATASET


## Training
  ```
  python3 CycleGAN.py --cuda --oridataPath ./** --bludataPath ./**
  ```

## Generate
  ```
  python3 generate.py --G_AB checkpoints/G_AB_6000.pth --G_BA checkpoints/G_BA_6000.pth -cuda --bludataPath ./** ordataPath ./**
  ```
To train or generate on dataset, change `dataPath` accordingly.
## Result
```
<img src=videotio.mp4" width="256"> 
```

## Reference
1. [https://github.com/junyanz/CycleGAN](https://github.com/junyanz/CycleGAN)
2. Zhu J Y, Park T, Isola P, et al. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks[J]. arXiv preprint arXiv:1703.10593, 2017.
3. [https://github.com/sunshineatnoon/Paper-Implementations/tree/master/cycleGAN](https://github.com/sunshineatnoon/Paper-Implementations/tree/master/cycleGAN)
