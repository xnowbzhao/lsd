# The structure of profile1.txt and profile2.txt

```
4                      <- Number of the ground truth meshes and noisy meshes
strain/cad.obj         <- Path of GT mesh
strain/elephant.obj
strain/octa-flower.obj
strain/fandisk.obj
strain/cad_n1.obj      <- Path of noisy mesh
strain/elephant_n2.obj
strain/octa-flower_n3.obj
strain/fandisk_n2.obj
dev/01t00t00.npy       <- Path and name of LSD files
dev/F_01t00t00.npy     <- Path and name of ground truth normal files
26 35 1 4000 1000  
   ↑
Parameters for train data number 
0,1,2: the index range of output files groups, range(10, 20, 2) = 10, 12, 14, 16, 18
3: the number of LSD in each group
4: the nnumber of LSD in each file 
```

# The structure of s_i2.txt

```
2                      <- Number of models
model/s_i1             <- Path of model
model/s_i2
2 20                   <- Number of Nf and Nv
75                     <- Number of noisy meshes
stest/armadillo_n1.obj <- Path of noisy mesh
stest/armadillo_n2.obj
stest/armadillo_n3.obj
stest/block_n1.obj
stest/block_n2.obj
stest/block_n3.obj
...

```

