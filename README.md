# Partially-Connected-Unrolled-RL-Network (PC-RLN)
Feng Tian and Weijian Yang, "Unrolled Richardson-Lucy deconvolution network with partially connected layers in computational microscopy," Opt. Express 34, 18676-18689 (2026)
### Clone this repository:
```
git clone https://github.com/Yang-Research-Laboratory/Partially-Connected-Unrolled-RL-Network
```

## [Paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-34-10-18676###)
![schematicimage](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/images/Figure1.png)<br />
## Training PC-RLN for Fourier light field microscopy (FLFM)
<p align="center">
  <img src="https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/images/Figure2.png" width="80%">
</p>

We provided the 3D point spread function ([**PSF**](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/Fourier%20Light%20Field%20Microscopy/PSF_FLFM.mat)) and [**data_FLFM.mat**](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/Fourier%20Light%20Field%20Microscopy/data_FLFM.mat) of our simulated FLFM system, with 60 training sample sets and 8 testing sample sets. Our FLFM dataset is adopted from Yi et al. ([**F-VCD**](https://www.nature.com/articles/s42003-023-05636-x)) [1].<br /><br />

Run MATLAB script [**buildindex_FLFM.m**](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/Fourier%20Light%20Field%20Microscopy/buildindex_FLFM.m) to extract pixel-voxel mapping for the given PSF and extract the voxel / pixel coords with initialized weights from PSF intensity to build the partically connected layers. The preextracted index mapping and inital weights are stored in [**data_FLFM_index.mat**](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/Fourier%20Light%20Field%20Microscopy/data_FLFM_index.mat).<br /><br />

Run [**PCRLNet_FLFM.py**](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/Fourier%20Light%20Field%20Microscopy/buildindex_FLFM.m) to construct the PC-RLN architecture based on FLFM PSF and perform training and testing on the datasets.

## Training PC-RLN for mask-based computational imager
<p align="center">
  <img src="https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/images/Figure3.png" width="70%">
</p>

We provided the simulation dataset [**data_MLA.m**](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/Microlens%20Array%20Microscopy/data_MLA.mat) and pixel-voxel indexing map [**data_MLA_index.mat**](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/Microlens%20Array%20Microscopy/data_MLA_index.mat) from a microlens array (MLA) imaging system with local (spatially varying) PSF. The training sample set is generated from randomly positioned / orinted alphabet letters across field of view. The object volume contains 13 depths.<br /><br />

Run [**PCRLNet_MLA.py**](https://github.com/Fengshub/Partially-Connected-Unrolled-RL-Network/blob/main/Microlens%20Array%20Microscopy/PCRLNet_MLA.py) to construct PC-RLN architecture based on MLA PSF and perform training and testing. <br /><br />


[1]. Yi, C., Zhu, L., Sun, J. et al. Video-rate 3D imaging of living cells using Fourier view-channel-depth light field microscopy. Commun Biol 6, 1259 (2023). https://doi.org/10.1038/s42003-023-05636-x
