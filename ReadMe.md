## Enhansing PBFL for ICLR 2025

### Dataset setting
```sh
cd dataset
python generate_cifar10.py noniid - dir
```

### PFL training using cwFedAvg algorithm
```sh
# FedProto with PPA
cd system
python main.py -did 0 -data Cifar10 -m Ht0 -algo FedProto -ppa

# FedProto with CPKD
cd system
python main.py -did 0 -data Cifar10 -m Ht0 -algo FedProto -cpkd 

# FedProto with CPS with 10%
cd system
python main.py -did 0 -data Cifar10 -m Ht0 -algo FedProto -cps -rcps 10

```
