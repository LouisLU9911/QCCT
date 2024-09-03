# QUIC Congestion Control Transformer (QCCT)

## Dataset

You can generate your dataset through:

```bash
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

```bash
# clone all
$ cd path/to/your/workspace
$ git clone https://github.com/LouisLU9911/quic-traffic-simulator.git

# build
$ make build

# docker-compose up
$ make

# apply tc
$ make tc
```
