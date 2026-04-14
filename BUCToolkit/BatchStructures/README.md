# Here `data.py` and `batch.py` are a subset from `pytorch_geometric`

BUCToolkit uses the most basic graph data structures of
[`pytorch_geometric`](https://pytorch-geometric.readthedocs.io/en/latest/) [1, 2] framework as a 
basic standard input data format.

Thus, one needs not install the whole `pytorch_geometric` with heavy dependencies to run our methods.

We follow the same approach to NequIP (https://github.com/mir-group/nequip/tree/main/nequip) and 
mace (https://github.com/ACEsuit/mace), and copy their code of `data.py` and `batch.py` here.


[1]  Fey, M., & Lenssen, J. E. (2019). Fast Graph Representation Learning with PyTorch Geometric (Version 2.0.1) [Computer software]. https://github.com/pyg-team/pytorch_geometric <br>
[2]  https://arxiv.org/abs/1903.02428