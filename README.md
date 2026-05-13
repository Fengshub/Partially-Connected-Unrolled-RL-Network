# Partially-Connected-Unrolled-RL-Network (PC-RLN)
Feng Tian and Weijian Yang, "Unrolled Richardson-Lucy deconvolution network with partially connected layers in computational microscopy," Opt. Express 34, 18676-18689 (2026)
### Clone this repository:
```
git clone https://github.com/Yang-Research-Laboratory/Partially-Connected-Unrolled-RL-Network
```

## [Paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-34-10-18676###)

## Training PC-RLN for Fourier light field microscopy (FLFM)
![schematicimage](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/images/Figure1.PNG)
We provided the 3D point spread function (PSF) of our simulated FLFM system, with 60 training sample sets and 8 testing sample sets. Our FLFM dataset is adopted from et. al. (F-VCD) [].
We included the script to extract pixel-voxel mapping for the given PSF and extract the voxel / pixel coords with initialized weights from PSF intensity to build the partically connected layers
We provided the .py script to construct the PC-RLN architecture based on FLFM PSF and perform training and testing on the datasets.
