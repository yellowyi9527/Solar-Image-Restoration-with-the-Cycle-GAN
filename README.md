# Solar-Image-Restoration-with-the-Cycle-GAN

## Prerequisites
- PyTorch
- torchvision

## DATASET
Data sets need to be built in advance according to the principles of the paper.

Save the high-resolution data to the new folder ./ori/

Save degraded data to the new folder ./blu/


## Training
  ```
  python3 CycleGAN.py --cuda --oridataPath ./ori --bludataPath ./blu  ```

## Generate
  ```
  python3 generate.py --G_AB checkpoints/G_AB_6000.pth --G_BA checkpoints/G_BA_6000.pth -cuda --bludataPath ./blu --ordataPath ./ori
  ```
To train or generate on dataset, change `dataPath` accordingly.
## Result
### Simulated Blurred Tio Data 
<img src="https://github.com/yellowyi9527/Solar-Image-Restoration-with-the-Cycle-GAN/blob/master/out_picture/tio.png" width="600px"/>

### Simulated Blurred  H-alpha Data
<img src="https://github.com/yellowyi9527/Solar-Image-Restoration-with-the-Cycle-GAN/blob/master/out_picture/halpha.png" width="600px"/>

### Real Observation Tio Video Data
<img src='https://github.com/yellowyi9527/Solar-Image-Restoration-with-the-Cycle-GAN/blob/master/out_picture/videoTio.gif'>

### Real Observation H-alpha Video Data
<img src='https://github.com/yellowyi9527/Solar-Image-Restoration-with-the-Cycle-GAN/blob/master/out_picture/videoHa.gif'>

## Reference
1. [https://github.com/junyanz/CycleGAN](https://github.com/junyanz/CycleGAN)
2. Zhu J Y, Park T, Isola P, et al. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks[J]. arXiv preprint arXiv:1703.10593, 2017.
3. [https://github.com/sunshineatnoon/Paper-Implementations/tree/master/cycleGAN](https://github.com/sunshineatnoon/Paper-Implementations/tree/master/cycleGAN)
