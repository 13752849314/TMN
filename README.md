# TMN: Transformer in Matrix Network for Single Image Super-Resolution with Enhanced Shallow Feature Preservation

## Environment

- pytorch>=2.0

### Quick start

```shell
git clone https://github.com/13752849314/TMN.git
cd TMN
```

### How to train

```shell
python train.py -opt ./config/MTN_x4.yml
```

### How to test

```shell
python test.py -opt ./config/MTN_x4.yml
```

Before training and testing, prepare the configuration file(*.yml)

```
@article{AO2025105207,
title = {TMN: Transformer in matrix network for single image super-resolution with enhanced shallow feature preservation},
journal = {Digital Signal Processing},
volume = {162},
pages = {105207},
year = {2025},
issn = {1051-2004},
doi = {https://doi.org/10.1016/j.dsp.2025.105207},
url = {https://www.sciencedirect.com/science/article/pii/S1051200425002295},
author = {Ou Ao and Zhenhong Shang},
keywords = {Image super-resolution, Deep learning, Transformer, Lightweight},
abstract = {Transformer-based image super-resolution has witnessed remarkable advancements in recent years. However, as transformer networks grow in depth, numerous existing super-resolution methods encounter challenges in effectively preserving shallow features, which play a crucial role in single image super-resolution. The low-resolution input image contains crucial structural and contextual information, and the shallow features serve as the carriers of this information. To address the challenge of preserving shallow features, we propose the Transformer in Matrix Network (TMN), a novel architecture specifically tailored for single image super-resolution. TMN incorporates a redesigned and optimized matrix mapping module, which arranges transformer blocks in a matrix structure to preserve and effectively exploit shallow features while facilitating the efficient reuse of hierarchical feature representations across the network. Additionally, TMN refines the efficient transformer to augment its capacity for modelling long-range dependencies, thereby enabling enhanced integration of information from spatially correlated regions within the image. To further enhance the reconstruction performance, TMN incorporates the structural loss into the loss function. By constraining the relevant statistical quantities, it improves the perceptual fidelity and preserves the intricate details. Experimental results show that TMN achieves competitive performance, with a reduction in computational costs by approximately one-third compared to leading methods like SwinIR. TMN's efficient design and high-quality reconstruction make it particularly suitable for deployment on resource-constrained devices, addressing a critical need in practical applications. The implementation code is publicly available at https://github.com/13752849314/TMN.}
}
```

