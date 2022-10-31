# GenPT
[Findings of EMNLP 2022] **[Generative Prompt Tuning for Relation Classification](https://arxiv.org/abs/2210.12435)**. 

### Requirements

- ``python 3.6``
- ``PyTorch 1.7.1``
- ``transformers 4.10.1``
- ``numpy 1.19.2``

#### Install dependencies
Please install all the dependency packages using the following command:
```bash
conda env create -f genpt.yaml
conda activate genpt
```

## Prepare the data

We experiment our model on four relation classification datasets,

  1. [TACRED](https://nlp.stanford.edu/projects/tacred/)
  2. [TACREV](https://github.com/DFKI-NLP/tacrev)
  3. [Re-TACRED](https://github.com/gstoica27/Re-TACRED)
  4. [Wiki80](https://github.com/thunlp/OpenNRE)

Please download data from the official links and put it under the ``./data/``.
The expected structure of files is:
```
GenPT
 |-- data
 |    |-- tacred
 |    |    |-- train.json        
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |    |-- rel2id.json
 |    |-- tacrev
 |    |    |-- ...
 |    |-- retacred
 |    |    |-- ...
 |    |-- wiki80
 |    |    |-- ...
```
Then use the following command to generate the few-shot data we need. Take the generation of k=8 on TACRED as an example,,
```bash
python code/generate_k_shot_data.py --data_dir data/tacred --output_dir data/tacred --k 8
```
The sampled few-shot data will be in `data/tacred/k-shot`.

## Training
To run our model, use command in the root directory

```bash
bash scripts/run_tacred_bart.sh
```
The experiments can be conducted with one NVIDIA Quadro RTX 8000 with 48GB memory.

## Cite
Please cite the following paper: "Generative Prompt Tuning for Relation Classification". Jiale Han, Shuai Zhao, Bo Cheng, Shengkun Ma, and Wei Lu. Findings of EMNLP, 2022.

