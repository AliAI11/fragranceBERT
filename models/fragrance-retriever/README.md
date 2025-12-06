---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:7000
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: romantic date night scent
  sentences:
  - 'aventus-oil by creed. accords: woody, fruity, mossy. top notes: pineapple, bergamot,
    black currant, apple. middle notes: birch, patchouli, jasmine, rose. base notes:
    oakmoss, musk, ambergris, vanilla.'
  - 'dark-wash by zara. accords: aquatic, woody, aromatic. top notes: sesame. middle
    notes: watery notes. base notes: sandalwood.'
  - 'miss-charming by oriflame. accords: white floral, citrus, lactonic. top notes:
    mandarin orange. middle notes: gardenia. base notes: vanilla.'
- source_sentence: boadicea-the-victorious dragon perfume
  sentences:
  - 'cortigiana by il-profvmo. accords: cherry, floral, sweet. top notes: cherry,
    wildflowers. middle notes: almond, vanilla. base notes: iris, herbal notes.'
  - 'dragon by boadicea-the-victorious. accords: amber, warm spicy, smoky. top notes:
    cinnamon, saffron. middle notes: incense, magnolia, amber. base notes: cashmere
    musk, vanilla.'
  - 'eclat-d-arpege-sheer by lanvin. accords: floral, white floral, fresh. top notes:
    lotus leaf, mandarin orange, pitahaya. middle notes: lily-of-the-valley, peony,
    water jasmine. base notes: white musk, cedar, amber.'
- source_sentence: date night perfume with vanilla
  sentences:
  - 'evolution-de-l-homme-matin by parfums-vintage. accords: citrus, aromatic, woody.
    top notes: grapefruit, lime, juniper berries, thyme, lemon, pink pepper, artemisia,
    bergamot. middle notes: patchouli, cedar, jasmine, black currant, apple, rose,
    cypriol oil or nagarmotha. base notes: ambergris, musk, birch, vanilla.'
  - 'giungle-di-seta by salvatore-ferragamo. accords: floral, musky, fresh. top notes:
    pea. middle notes: peony. base notes: musk.'
  - 'pink-bouquet by moschino. accords: fruity, sweet, fresh. top notes: raspberry,
    pineapple, bergamot. middle notes: peony, lily-of-the-valley, violet, jasmine.
    base notes: gingerbread, peach, musk, oakmoss.'
- source_sentence: perfume with orange blossom and rose
  sentences:
  - 'alatau by faberlic. accords: green, aromatic, woody. top notes: water notes,
    mint, amalfi lemon, red berries. middle notes: hyacinth, lily-of-the-valley, violet,
    orange blossom, wild rose. base notes: pine, oak moss, musk, vetiver, orris root,
    amber, patchouli.'
  - 'perle-imperiale by guerlain. accords: powdery, woody, fruity. top notes: bergamot.
    middle notes: fig, powdery notes. base notes: sandalwood, leather, myrrh.'
  - 'aire-sensual by loewe. accords: citrus, white floral, musky. top notes: lemon,
    petitgrain, green apple, mandarin orange. middle notes: freesia, lily-of-the-valley,
    jasmine. base notes: musk, vetiver, cedar, amber.'
- source_sentence: byredo mumbai-noise for winter
  sentences:
  - 'oud-loukoum by bortnikoff. accords: woody, oud, yellow floral. top notes: dried
    fruits, tobacco. middle notes: ylang-ylang. base notes: peru balsam, agarwood
    (oud), guaiac wood, cedar, indian oud.'
  - 'mumbai-noise by byredo. accords: oud, aromatic, amber. top notes: davana. middle
    notes: coffee, tonka bean. base notes: agarwood (oud), sandalwood, labdanum.'
  - 'cafe-pop by cafe-parfums. accords: floral, fresh, white floral. top notes: pear,
    freesia, black currant. middle notes: peony, jasmine, lily-of-the-valley. base
    notes: musk, sandalwood, amber.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: val
      type: val
    metrics:
    - type: cosine_accuracy@1
      value: 0.03133333333333333
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.10933333333333334
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.18
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.38333333333333336
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.03133333333333333
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.03644444444444444
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.036000000000000004
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.03833333333333333
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.03133333333333333
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.10933333333333334
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.18
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.38333333333333336
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.16924391896197283
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.10619047619047602
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.1171259731007836
      name: Cosine Map@100
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'byredo mumbai-noise for winter',
    'mumbai-noise by byredo. accords: oud, aromatic, amber. top notes: davana. middle notes: coffee, tonka bean. base notes: agarwood (oud), sandalwood, labdanum.',
    'cafe-pop by cafe-parfums. accords: floral, fresh, white floral. top notes: pear, freesia, black currant. middle notes: peony, jasmine, lily-of-the-valley. base notes: musk, sandalwood, amber.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.4051, 0.0044],
#         [0.4051, 1.0000, 0.0592],
#         [0.0044, 0.0592, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `val`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.0313     |
| cosine_accuracy@3   | 0.1093     |
| cosine_accuracy@5   | 0.18       |
| cosine_accuracy@10  | 0.3833     |
| cosine_precision@1  | 0.0313     |
| cosine_precision@3  | 0.0364     |
| cosine_precision@5  | 0.036      |
| cosine_precision@10 | 0.0383     |
| cosine_recall@1     | 0.0313     |
| cosine_recall@3     | 0.1093     |
| cosine_recall@5     | 0.18       |
| cosine_recall@10    | 0.3833     |
| **cosine_ndcg@10**  | **0.1692** |
| cosine_mrr@10       | 0.1062     |
| cosine_map@100      | 0.1171     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 7,000 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                          |
  |:--------|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                              |
  | details | <ul><li>min: 4 tokens</li><li>mean: 8.03 tokens</li><li>max: 18 tokens</li></ul> | <ul><li>min: 36 tokens</li><li>mean: 60.92 tokens</li><li>max: 114 tokens</li></ul> |
* Samples:
  | sentence_0                                          | sentence_1                                                                                                                                                                                                                                                          |
  |:----------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>floral green and fruity perfume</code>        | <code>julia by teo-cabanel. accords: fruity, green, floral. top notes: rhuburb, black currant, mandarin orange. middle notes: hyacinth, violet, jasmine. base notes: raspberry, musk, sandalwood, labdanum, incense.</code>                                         |
  | <code>faberlic perfume for romantic evenings</code> | <code>vent-d-aventures-pour-femme by faberlic. accords: fruity, white floral, ozonic. top notes: red apple, watermelon, citron, black currant. middle notes: lily, hyacinth, rose, violet, galbanum, plum. base notes: cedar, amber, tonka bean, white musk.</code> |
  | <code>best perfume for a winter date</code>         | <code>miss-soprani by luciano-soprani. accords: floral, white floral, fresh. top notes: peony, guava, freesia, ylang-ylang, mandarin orange. middle notes: jasmine, magnolia. base notes: amber, oakmoss, vetiver.</code>                                           |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss | val_cosine_ndcg@10 |
|:------:|:----:|:-------------:|:------------------:|
| 1.0    | 438  | -             | 0.1663             |
| 1.1416 | 500  | 1.3625        | 0.1631             |
| 2.0    | 876  | -             | 0.1684             |
| 2.2831 | 1000 | 1.1209        | 0.1668             |
| 3.0    | 1314 | -             | 0.1692             |


### Framework Versions
- Python: 3.12.12
- Sentence Transformers: 5.1.2
- Transformers: 4.57.2
- PyTorch: 2.9.0+cu126
- Accelerate: 1.12.0
- Datasets: 4.0.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->