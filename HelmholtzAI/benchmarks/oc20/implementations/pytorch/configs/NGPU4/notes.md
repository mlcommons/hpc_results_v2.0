# Notes on config files

## Directory and include structure
- `NGPU4`/`NGPU8` for nodes with 4/8 GPUs
- `strong`/`weak` for strong and weak scaling configs
- all configs in NGPU4 include `NGPU4/common.sh` which
	- includes the base config `config_DGXA100_common.sh`
	- and overwrites the number of GPUs (4) and the eval batch size (32)

## Naming convention
- for strong scaling: `config_<GPU>_<number of nodes>x<number of GPUs per node>x<local batch size>.sh`
- for weak scaling: `config_<GPU>_<number of instances>x<number of nodes_per_instance>x<number of GPUs per node>x<local batch size>.sh`

## Batch Size
- Maximum global batch size used by original configs: 2048 (256 nodes, 8 GPUs per node, local batch size 1)
- Limitations due to memory constraints on A100 40GB GPUs
	- maximum local batch size: 16? (32 results in CUDA out of memory error)
	- maximum local evaluation batch size: 32