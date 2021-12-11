
[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

Overview
------

* This is a PyTorch implementation of the paper [**Interpretable and Lightweight 3-D Deep Learning Model For Automated ACL Diagnosis**](https://ieeexplore.ieee.org/document/9435063) by Jeon et al.

* Paper DOI: 10.1109/JBHI.2021.3081355



### Prerequisites

The software is developed in **Python 3.7+**. For the deep learning, the **PyTorch 1.3.1+** framework is used.



Code structure
---
1. Everything can be ran from *./main_ACL.py*. 
2. The data preprocessing parameters, hyper-parameters, model parameters, and directories can be modified from *./config/config.yaml*.
* Also, you should first choose an `experiment` name (if you are starting a new experiment) for training, in which all the evaluation and loss value statistics, tensorboard events, and model & checkpoints will be stored. Furthermore, a `config.yaml` file will be created for each experiment storing all the information needed.
* For testing, just load the experiment which its model you need.

3. The rest of the files:
* *./models/* directory contains all the model architectures.
* *./Train_Valid_ACL.py* contains the training and validation processes.

