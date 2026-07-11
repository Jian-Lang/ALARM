#! /bin/bash
# Stage 1: Confidence-based Explicit Meme Identification (pseudo-label the explicit set)
python src/main.py --config-name label_qwen_ToxiCN
# Stage 2: Multimodal Retrieval-based Contrastive Pairing (embeddings + contrastive pairs)
python src/model/Experience/make_embeddings.py
python src/model/Experience/conduct_retrieval.py
# Stage 3: Pairwise Learning-guided Experience Gathering
python src/main.py --config-name experience_qwen_ToxiCN
# Stage 4: Reference Refinement
python src/main.py --config-name reference_qwen_ToxiCN
# Stage 5: Inference (predict the subtle set with the self-generated references)
python src/main.py --config-name inpredict_qwen_ToxiCN
