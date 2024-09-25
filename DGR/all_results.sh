#!/usr/bin/env bash



python3 compare_FI.py --seed=2 --n-seeds=10 --experiment=splitMNIST --scenario=task
python3 compare_FI.py --seed=2 --n-seeds=5 --experiment=splitMNIST --scenario=domain
python3 compare_FI.py --seed=2 --n-seeds=5 --experiment=splitMNIST --scenario=class
python3 compare_FI.py --seed=2 --n-seeds=10 --experiment=splitMNIST --scenario=task --online
python3 compare_FI.py --seed=2 --n-seeds=10 --experiment=splitMNIST --scenario=task --online --fisher-n-all=1000
python3 compare_FI.py --seed=2 --n-seeds=5 --experiment=splitMNIST --scenario=task --online --precondition
python3 compare_FI.py --seed=2 --n-seeds=2 --experiment=splitMNIST --scenario=task --online --precondition --alpha=1e-4
#
python3 compare_FI.py --seed=2 --n-seeds=3 --experiment=splitMNIST --scenario=class --online --precondition
python3 compare_preconditioning.py --seed=2 --n-seeds=3 --experiment=splitMNIST --scenario=task --fisher-n=500
python3 compare_preconditioning.py --seed=2 --n-seeds=3 --experiment=splitMNIST --scenario=task
python3 compare_FI_KFAC.py --seed=2 --n-seeds=10 --experiment=splitMNIST --scenario=task
python3 compare_FI_KFAC.py --seed=2 --n-seeds=5 --experiment=splitMNIST --scenario=task --precondition
python3 compare_FI_rand.py --seed=2 --n-seeds=5 --experiment=splitMNIST --scenario=task --online

python3 compare_FI.py --seed=2 --n-seeds=2 --experiment=permMNIST --contexts=10 --iters=2000 --lr=0.001 --scenario=domain --online --no-all
python3 compare_FI.py --seed=2 --n-seeds=2 --experiment=permMNIST --contexts=10 --iters=2000 --lr=0.001 --scenario=domain --online --precondition --no-all
python3 compare_preconditioning.py --seed=2 --n-seeds=2 --experiment=permMNIST --contexts=10 --iters=2000 --lr=0.001 --scenario=domain --fisher-n=500
python3 compare_preconditioning.py --seed=2 --n-seeds=2 --experiment=permMNIST --contexts=10 --iters=2000 --lr=0.001 --scenario=domain --fisher-labels=true
python3 compare_preconditioning.py --seed=2 --n-seeds=2 --experiment=permMNIST --contexts=10 --iters=2000 --lr=0.001 --scenario=domain --fisher-labels=true --fisher-batch=128

python3 compare_FI.py --seed=2 --n-seeds=5 --experiment=CIFAR10 --contexts=5 --conv-type=resNet --fc-layers=1 --iters=2000 --reducing-layers=3 --depth=5 --global-pooling --channels=20 --lr=0.001 --scenario=task
python3 compare_FI.py --seed=2 --n-seeds=10 --experiment=CIFAR10 --contexts=5 --conv-type=resNet --fc-layers=1 --iters=2000 --reducing-layers=3 --depth=5 --global-pooling --channels=20 --lr=0.001 --scenario=task --online
python3 compare_FI.py --seed=2 --n-seeds=5 --experiment=CIFAR10 --contexts=5 --conv-type=resNet --fc-layers=1 --iters=2000 --reducing-layers=3 --depth=5 --global-pooling --channels=20 --lr=0.001 --scenario=task --online --precondition
python3 compare_preconditioning.py --seed=2 --n-seeds=5 --experiment=CIFAR10 --contexts=5 --conv-type=resNet --fc-layers=1 --iters=2000 --reducing-layers=3 --depth=5 --global-pooling --channels=20 --lr=0.001 --scenario=task
python3 compare_preconditioning.py --seed=2 --n-seeds=5 --experiment=CIFAR10 --contexts=5 --conv-type=resNet --fc-layers=1 --iters=2000 --reducing-layers=3 --depth=5 --global-pooling --channels=20 --lr=0.001 --scenario=task --fisher-n=500
python3 compare_preconditioning.py --seed=2 --n-seeds=5 --experiment=CIFAR10 --contexts=5 --conv-type=resNet --fc-layers=1 --iters=2000 --reducing-layers=3 --depth=5 --global-pooling --channels=20 --lr=0.001 --scenario=task --fisher-labels=true
python3 compare_preconditioning.py --seed=2 --n-seeds=5 --experiment=CIFAR10 --contexts=5 --conv-type=resNet --fc-layers=1 --iters=2000 --reducing-layers=3 --depth=5 --global-pooling --channels=20 --lr=0.001 --scenario=task --fisher-labels=true --fisher-batch=128

python3 compare_FI.py --seed=2 --n-seeds=3 --experiment=CIFAR100 --scenario=task --pre-convE --freeze-convE --seed-to-ltag
python3 compare_FI.py --seed=2 --n-seeds=3 --experiment=CIFAR100 --scenario=domain --pre-convE --freeze-convE --seed-to-ltag
#python3 compare_FI.py --seed=2 --n-seeds=1 --experiment=CIFAR100 --scenario=class --pre-convE --freeze-convE --seed-to-ltag
python3 compare_FI.py --seed=2 --n-seeds=1 --experiment=CIFAR100 --scenario=task --pre-convE --freeze-convE --seed-to-ltag --online
python3 compare_FI.py --seed=2 --n-seeds=1 --experiment=CIFAR100 --scenario=task --pre-convE --freeze-convE --seed-to-ltag --online --precondition
python3 compare_preconditioning.py --seed=2 --n-seeds=1 --experiment=CIFAR100 --scenario=task --pre-convE --freeze-convE --seed-to-ltag --fisher-n=500
python3 compare_preconditioning.py --seed=2 --n-seeds=1 --experiment=CIFAR100 --scenario=task --pre-convE --freeze-convE --seed-to-ltag --fisher-labels=true
python3 compare_preconditioning.py --seed=2 --n-seeds=1 --experiment=CIFAR100 --scenario=task --pre-convE --freeze-convE --seed-to-ltag --fisher-labels=true --fisher-batch=128




## MNIST

./compare_hyperParams.py --seed=1 --experiment=splitMNIST --scenario=task
./compare_hyperParams.py --seed=1 --experiment=splitMNIST --scenario=domain
./compare_hyperParams.py --seed=1 --experiment=splitMNIST --scenario=class

./compare.py --seed=2 --n-seeds=20 --experiment=splitMNIST --scenario=task
./compare.py --seed=2 --n-seeds=20 --experiment=splitMNIST --scenario=domain
./compare.py --seed=2 --n-seeds=20 --experiment=splitMNIST --scenario=class

./compare_replay.py --seed=2 --n-seeds=5 --experiment=splitMNIST --scenario=task --tau-per-budget
./compare_replay.py --seed=2 --n-seeds=5 --experiment=splitMNIST --scenario=domain --tau-per-budget
./compare_replay.py --seed=2 --n-seeds=5 --experiment=splitMNIST --scenario=class --tau-per-budget


## Task-free Split MNIST

./compare_hyperParams_task_free.py --seed=1 --experiment=splitMNIST --scenario=task
./compare_hyperParams_task_free.py --seed=1 --experiment=splitMNIST --scenario=domain
./compare_hyperParams_task_free.py --seed=1 --experiment=splitMNIST --scenario=class

./compare_task_free.py --seed=2 --n-seeds=20 --experiment=splitMNIST --scenario=task --gating-prop=0.45 --c=10.
./compare_task_free.py --seed=2 --n-seeds=20 --experiment=splitMNIST --scenario=domain --c=10.
./compare_task_free.py --seed=2 --n-seeds=20 --experiment=splitMNIST --scenario=class --c=10.

#