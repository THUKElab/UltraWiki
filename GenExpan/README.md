# 🏝️ GenExpan



## 🔬 Dependencies

```shell
pip install -r requirements.txt
```

#### Details

- python==3.11.7
- pytorch==2.1.2
- transformers==4.36.2

## 📚 File hierarchy

- **File hierarchy**

```
GenExpan
├── beam_search.py
├── CoT.py
├── ds_config.json
├── EntityTrie.py
├── Expan.py
├── main.py
├── make_sentences.py
├── README.md
├── requirements.txt
├── run_base.sh
├── run_cot.sh
├── run_ra.sh
├── summarize.py
├── train_lm.py
├── train_lm.sh
└── utils.py

```



## 🚀 Train and Evaluate

---

- The `train_lm.sh` script is used for training models. The training checkpoints will be saved in the `train_output` directory. Before expanding entities, you need to train the model using a corpus. Just run this:

```shell
bash train_lm.sh
```



- `run_base.sh`, `run_cot.sh`, and `run_ra.sh` are respectively the running scripts for three methods: ***GenExpan***, ***GenExpan with Chain-of-thought Reasoning***, and ***GenExpan with Entity-based Retrieval Augmentation***. Their corresponding relationships are shown in the following table:

| Script Name |                      Method                       |
| :---------- | :-----------------------------------------------: |
| run_base.sh |                     GenExpan                      |
| run_cot.sh  |     GenExpan with Chain-of-thought Reasoning      |
| run_ra.sh   | GenExpan with Entity-based Retrieval Augmentation |



- We use 6 A100 GPUs with 80GB of VRAM each for training and inference. In fact, the inference phase of  ***GenExpan*** and ***GenExpan with Chain-of-thought Reasoning*** can also be performed on 24GB VRAM RTX 3090.



- If you want to expand entities with ***GenExpan***, run this:

```shell
bash run_base.sh
```

The expand results will be saved in `../data/expand_results_GenExpan_base` .



- If you want to expand entities with ***GenExpan with Chain-of-thought Reasoning***, run this:

```shell
bash run_cot.sh
```

The expand results will be saved in `../data/expand_results_GenExpan_cot` .



- If you want to expand entities with ***GenExpan with Entity-based Retrieval Augmentation***,  run this:

```shell
bash run_ra.sh
```

The expand results will be saved in `./data/expand_results_GenExpan_ra` .





## 💡 Acknowledgement

- We appreciate  [```ProbExpan```](https://github.com/geekjuruo/ProbExpan) , [`MESED`](https://github.com/THUKElab/MESED) and many other related works for their open-source contributions.

