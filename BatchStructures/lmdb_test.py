# -*- coding: utf-8 -*-
import lmdb
from Preprocessing.load_files import POSCARs2Feat
import numpy as np
import pickle as pkl

data = POSCARs2Feat('/home/ppx/PythonProjects/DataBases/new_MP_all_data', 2)
data.read()
# map_size定义最大储存容量，单位是kb，以下定义1TB容量
env = lmdb.open("./example", map_size=1024*1024*100)

txn = env.begin(write=True)

# 添加数据和键值
txn.put(key='1'.encode(), value=pkl.dumps(data))
txn.put(key='2'.encode(), value='bbb'.encode())
txn.put(key='3'.encode(), value='ccc'.encode())

# 修改数据
txn.put(key='3'.encode(), value='ddd'.encode())

# 通过commit()函数提交更改
txn.commit()
env.close()