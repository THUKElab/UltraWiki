# 🏝️ UltraWiki: Ultra-fine-grained Entity Set Expansion with Negative Seed Entities



## 🔬 Dependencies

```bash
pip install -r requirements.txt
```

#### Details

- python==3.11
- pytorch==2.2.1
- transformers==4.38.2
- openai==0.27.8

## 📚 Dataset(UntraWiki)

- download query from https://cloud.tsinghua.edu.cn/d/811f767164994c268679/ and put them into "./data"

-  **File hierarchy**

```
UntraWiki
├── data
│   ├── query
│   │   ├── cls_1.json
│   │   ├── cls_2.json
│   │   ├── ...
│   │   └── cls_T.json
│   │   
│   ├── ent2sents.json
│   ├── ent2text.json
│   └── entities.txt
│
├── GenExpan
│
├── src
│   ├── dataset_for_cl.py
│   ├── dataset_for_ent_predict.py
│   ├── expand.py
│   ├── inferencer.py
│   ├── main.py
│   ├── make_cln2groups.py
│   ├── make_ent2ids.py
│   ├── model.py
│   ├── train_mlm.py
│   └── utils.py
│
├── appendix.pdf
├── README.md
├── requirements.txt
├── run_base.sh
├── run_cl.sh
└── run_ra.sh

```



## 🚀 Train and Evaluate

---

- `run_base.sh` `run_cl.sh` and `run_ra.sh` are respectively the running scripts for three methods: ***RetExpan***, ***RetExpan with Ultra-fine-grained Contrastive Learning***, and ***RetExpan with Entity-based Retrieval Augmentation***. Their corresponding relationships are shown in the following table:

| Script Name |                        Method                         |
| :---------- | :---------------------------------------------------: |
| run_base.sh |                       RetExpan                        |
| run_cl.sh   | RetExpan with Ultra-fine-grained Contrastive Learning |
| run_ra.sh   |   RetExpan with Entity-based Retrieval Augmentation   |



- We use 8 RTX 3090 GPUs with 24GB of VRAM each for training and inference. In the `run*.sh` script, we set the GPU usage through `gpu_groups="0,1,2,3,4,5,6,7"`.



- If you want to expand entities with ***RetExpan***, run this:

```
bash run_base.sh
```

The expand results will be saved in ***./data/expand_results_base*** .



- If you want to expand entities with ***RetExpan with Ultra-fine-grained Contrastive Learning***, run this:

```
bash run_cl.sh
```

The expand results will be saved in ***./data/expand_results_cl2*** .



- If you want to expand entities with ***RetExpan with Entity-based Retrieval Augmentation***,  run this:

```
bash run_ra.sh
```

The expand results will be saved in ***./data/expand_results_ra*** .





## 💡 Acknowledgement

- We appreciate  [```ProbExpan```](https://github.com/geekjuruo/ProbExpan) , [`MESED`](https://github.com/THUKElab/MESED) and many other related works for their open-source contributions.

