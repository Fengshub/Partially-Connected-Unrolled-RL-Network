%%
clear;close all;clc;
%% --- Define image and object dimensions ---
psfh=384;psfw=384;
objh=160;objw=160;zc=7;
%% --- Load simulated 3D PSF ---
load('PSF_FLFM.mat');
psf=zeros(psfh,psfw,zc);
Hs=H;
H=zeros(768,768,zc);
for did=1:zc
    H(:,:,did)=full(Hs{did});
end
H=H/max(H(:));
%% --- Threhold PSF for sparse pixel-voxel mapping ---
psfth=0.8; % keep PSF pixels above 80% peak intensity for each PSF cluster.
for did=1:zc
    Htemp=H(:,:,did);%full(H{did});
    Htemp=imresize(Htemp,[psfh,psfw]);
    psftemp=zeros(size(Htemp));
    CC = bwconncomp(Htemp,26);
    for idx=1:CC.NumObjects
        temp=CC.PixelIdxList{idx};
        a=find(Htemp==max(Htemp(temp)));
        psftemp(a)=Htemp(a);
    end
    Htemp(Htemp<psfth*max(Htemp(:)))=0;
    Htemp=max(Htemp,psftemp);
    Htemp=Htemp/sum(Htemp(:));
    psf(:,:,did)=Htemp;%uint8(logical(Htemp));

end
[cx0, cy0, cz0] = ind2sub(size(psf), find(psf ~= 0));
%% --- List-RL system matrix and pixel-voxel mapping
orx=objh/2;ory=objw/2; % object center coordinate
dsr=1; % downsampling rate
Nxo=objh;Nyo=objw; % number of object voxels along lateral dimensions
Nx=psfh;Ny=psfw; % number of image pixels along lateral dimensions
index=1; % starting index of pixel-voxel pair
voxel_coords=zeros(round(zc*Nxo*Nyo*length(cx0)/1e6),1); % vectorized voxel coordinates for each pair
pixel_coords=zeros(round(Nx*Ny*length(cx0)/1e6),1); % vectorized pixel coordinates for each pair
F_weights_init=zeros(round(Nx*Ny*length(cx0)/1e6),1); % initialized forward PC layer weights
B_weights_init=zeros(round(Nx*Ny*length(cx0)/1e6),1); % initialized backward PC layer weights
for ridx=1:zc % scan object voxels
    disp(ridx)
    for idx1=1:dsr:objh % scan object voxels
        iidx1=ceil(idx1/dsr);
        for idx2=1:dsr:objw % scan object voxels
            iidx2=ceil(idx2/dsr);
            cid_list=find(cz0==ridx);
            for idx3=1:length(cid_list) % scan mapped PSF pixels for current voxel
                cx=cx0(cid_list(idx3))+idx1-orx-1;
                cy=cy0(cid_list(idx3))+idx2-ory-1;
                if (cy>=1) && (cx>=1) && (cx<=Nx) && (cy<=Ny) % record pixel-voxel pairs and initial layer weights
                    voxel_coords(index,:)=((iidx1-1) * Nyo + (iidx2-1)) * zc + (ridx-1);
                    pixel_coords(index,:)=(cx-1)*Ny+(cy-1);
                    F_weights_init(index,:)=psf(cx0(cid_list(idx3)),cy0(cid_list(idx3)),cz0(cid_list(idx3)));
                    B_weights_init(index,:)=psf(cx0(cid_list(idx3)),cy0(cid_list(idx3)),cz0(cid_list(idx3)));
                    index=index+1;
                end
            end
        end
    end
end
voxel_coords=voxel_coords(1:index-1,:);
pixel_coords=pixel_coords(1:index-1,:);
F_weights_init=F_weights_init(1:index-1,:);
B_weights_init=B_weights_init(1:index-1,:);
%%
save('FLFM_index.mat','voxel_coords','pixel_coords','F_weights_init','B_weights_init')

    