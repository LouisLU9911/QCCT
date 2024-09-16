# QUIC Congestion Control Transformer (QCCT)

QUIC Congestion Control Transformer (QCCT) is a custom QUIC congestion controller using a transformer-like model to predict the next congestion window.

## Environment

* conda and python

```bash
$ export WORKSPACE=$(pwd)
$ git clone https://github.com/LouisLU9911/QCCT.git

$ cd $WORKSPACE/QCCT
# CPU
$ conda create --name quic python=3.10
$ conda activate quic
$ pip install -r requirements-cpu.txt
$ conda deactivate

# GPU
$ conda create --name quic-gpu python=3.10
$ conda activate quic-gpu
$ pip install -r requirements-gpu.txt
$ conda deactivate
```

* Rust and Cargo

## Dataset

You can generate your dataset through:

```bash
$ cd $WORKSPACE
$ git clone -b dev --single-branch https://github.com/LouisLU9911/s2n-quic.git
$ cd s2n-quic/quic/s2n-quic-sim

# generate dataset using seeds 42,2023,2024,10086
$ python gen_dataset.py --seeds 42,2023,2024,10086
$ ls | grep reports
reports_seed_10086/
reports_seed_2023/
reports_seed_2024/
reports_seed_42/
$ tree reports_seed_42
reports_seed_42
├── delay_100ms_drop_0.01
│   ├── context.json
│   ├── formatted.csv
│   ├── plan.toml
│   └── stderr.log
├── delay_200ms_drop_0.05
│   ├── context.json
│   ├── formatted.csv
│   ├── plan.toml
│   └── stderr.log
└── delay_500ms_drop_0.1
    ├── context.json
    ├── formatted.csv
    ├── plan.toml
    └── stderr.log
```

## Train and Test

```bash
$ cd $WORKSPACE/s2n-quic/quic/s2n-quic-sim
$ conda activate quic-gpu
$ python main.py
```

## Build QUIC server and client with QCCT

### QUIC server

```bash
$ conda deactivate && conda activate quic
$ cd $WORKSPACE/QCCT/custom-congestion-controller
$ cargo build
...
# Run server
$ ./target/debug/main
# if client connects...
Connection accepted from Ok(127.0.0.1:53766)
Stream opened from Ok(127.0.0.1:53766)
...
```

### QUIC client

```bash
$ cd $WORKSPACE/QCCT/echo
$ cargo build
...
$ ./target/debug/quic_echo_client
# now you can input something on the console
...
```

### Use a new model

If you want to use a new model:

```bash
$ cd $WORKSPACE
$ cp ./s2n-quic/quic/s2n-quic-sim/checkpoints/model_cpu_19.pt ./QCCT/custom-congestion-controller/model_cpu.pt
# NOTE: Don't forget to update the context_size in the lib.rs
$ vim QCCT/custom-congestion-controller/src/lib.rs
...
# rebuild and run
$ cargo build
...
$ ./target/debug/main
...
```
