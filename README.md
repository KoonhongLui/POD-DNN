# POD-DNN

We propose the POD-DNN, a novel algorithm leveraging deep neural networks (DNN) along with radial basis functions (RBF) in the context of the proper orthogonal decomposition (POD) reduced basis method (RBM), aimed at approximating the parametric mapping of parametric partial differential equations on irregular domains. The POD-DNN algorithm capitalizes on the low-dimensional characteristics of the solution manifold for parametric equations, alongside the inherent offline-online computational strategy of RBM and DNNs. In numerical experiments, POD-DNN demonstrates significantly accelerated computation speeds during the online phase. Compared to other algorithms that utilize RBF without integrating DNNs, POD-DNN substantially improves the computational speed in the online inference process. Furthermore, under reasonable assumptions, we have rigorously derived upper bounds on the complexity of approximating parametric mappings with POD-DNN, thereby providing a theoretical analysis of the algorithm's empirical performance.

[https://arxiv.org/abs/2404.06834](https://arxiv.org/abs/2404.06834)

# Requirements
The code mainly depends on [PyTorch](https://pytorch.org/). We also use the Python package [RBF](https://github.com/treverhines/RBF) for radial basis function (RBF) applications.

# Description
* [PointCloud.mat](./PointCloud.mat) contains the discrete points on the irregular domain, see Fig. 1 of our paper.
* [Dataset_Input_mat.npy](./Dataset_Input_mat.npy) and [Dataset_Output_mat.npy](Dataset_Output_mat.npy) are the datasets for neural network training, which can be generated by [Gen_Data.py](./Gen_Data.py). This program also generates the snapshot matrix [Snapshot_mat.npy](./Snapshot_mat.npy) and POD basis [POD_basis.npy](./POD_basis.npy). See Subsection 5.1 of our paper. 
* [Gen_RB.py](./Gen_RB.py) generates the reduced basis by the greedy algorithm LS-R $^2$ BFM described in the paper [Chen, Y., Gottlieb, S., Heryudono, A., Narayan, A.: A reduced radial basis function method for partial differential equations on irregular domains. J. Sci. Comput. 66(1), 67–90 (2016).](https://doi.org/10.1007/s10915-015-0013-8)
* [NN_LS.py](./NN_LS.py) trains the network and compares the numerical performance with LS-R $^2$ BFM. See Subsection 5.3 of our paper.
