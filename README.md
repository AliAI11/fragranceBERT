# 🌸 fragranceBERT
**Semantic Fragrance Search**  

Search 24,000+ fragrances the way you actually think about scent.

Try the Demo : https://huggingface.co/spaces/aali11/fragranceBERT
![demo](https://huggingface.co/spaces/aali11/fragranceBERT/resolve/main/demo.gif)  
## What is fragranceBERT?
A perfume search engine that understands how people actually describe scents.

Find a fragrance by writing what you feel:  
- cozy vanilla winter evenings  
- fresh citrus spring morning  
- dark rose in the rain  
- clean professional office scent  
- sexy leather date night  

Powered by a fine-tuned Sentence-BERT model trained on 10,000 synthetic natural-language queries.

## Performance
| Model                        | P@5   | P@10  | MRR   | NDCG@10 |
|------------------------------|-------|-------|-------|---------|
| **fragranceBERT (this project)** | **0.047** | **0.028** | **0.192** | **0.211** |
| TF-IDF baseline              | 0.030 | 0.020 | 0.104 | 0.125   |
| Pre-trained MiniLM (no FT)   | 0.026 | 0.015 | 0.106 | 0.116   |
| Keyword matching             | 0.001 | 0.001 | 0.005 | 0.005   |
| Random                       | 0.000 | 0.000 | 0.000 | 0.000   |

**+55–85%** better than TF-IDF  
**+80%** better than the same model without fine-tuning

## Quick Start
git clone https://github.com/aali11/fragranceBERT.git

cd fragranceBERT/notebooks

pip install sentence-transformers gradio numpy pandas scikit-learn

jupyter notebook gradio_demo.ipynb

Run all cells in the notebook to launch the search interface.


## Technical Details
Dataset: 24,063 perfumes from Fragrantica (Kaggle)

Synthetic queries: Qwen2.5-7B-Instruct

Base model: sentence-transformers/all-MiniLM-L6-v2

Loss: MultipleNegativesRankingLoss

Inference: Pre-computed embeddings + cosine similarity

*Built as a final project for Virginia Tech CS 5804 – Machine Learning (Fall 2025)*
