# LSD

【Pytorch Implementation of AAAI 2022 paper】 

Local Surface Descriptor for Geometry and Feature Preserved Mesh Denoising

The original network is trained with Tensorflow, for easy reproduction, I rewrite it with Pytorch.

## Environment

Openmesh 8.1

Eigen 3

CUDA 11.4

Pytorch 1.10.1

## Compilation

Run the following command to compiling LSD-denoising_mt.cpp and LSD-Gdata_mt.cpp, you may need to manually modify the path of OpenMesh-8.1 and Eigen.

```
export LD_LIBRARY_PATH=/***/OpenMesh-8.1/build/Build/lib:$LD_LIBRARY_PATH
g++ -std=c++11 LSD-denoising_mt.cpp libOpenMeshCore.so.8.1 -I ./ -I ./OpenMesh-8.1/src -O2 -o LSD-denoising_mt -pthread
g++ -std=c++11 LSD-Gdata_mt.cpp libOpenMeshCore.so.8.1 -I ./ -I ./OpenMesh-8.1/src -O2 -o LSD-Gdata_mt -pthread
```

## Generate Dataset and Training 

We take the synthetic dataset as an example to illustrate the training process and evaluation.

1, Download the dataset from [CNR](https://wang-ps.github.io/denoising.html), move the files as follow:

```
Synthetic\train\noisy -> strain
Synthetic\train\original -> strain
Synthetic\test\noisy -> stest
```

2, run the following commands for building training data. Since we found that on pytorch, the code needs more training data to achieve simialer results. We increased the number of training data:

```
./LSD-Gdata_mt profile/s_i1/profile1.txt
./LSD-Gdata_mt profile/s_i1/profile2.txt
```

3, train with the dataset to obtain the model of iteration 1, and move the model: 

```
python train.py
cp -r out model/s_i1
```

4,  denoise the training meshes with the model of iteration 1 to obtain the training meshes for iteration 2:

```
./LSD-denoising_mt profile/s_i1/s_i1.txt
```

5, repeat step 2-3 to obtain the model of iteration 2, notice that you should use the profile in profile/s_i2.

## Evaluation

Run the following command to obtain the denoised test meshes:
```
./LSD-denoising_mt profile/s_i2/s_i2.txt
```

## Citation
If the code is useful for your research, please consider citing:
  
    @inproceedings{zhao2022local,
      title={Local Surface Descriptor for Geometry and Feature Preserved Mesh Denoising},
      author={Zhao, Wenbo and Liu, Xianming and Jiang, Junjun and Zhao, Debin and Li, Ge and Ji, Xiangyang},
      year={2022}
    }

## Acknowledgement
The code is based on [GNF](https://github.com/bldeng/GuidedDenoising), If you use any of this code, please make sure to cite these works.

