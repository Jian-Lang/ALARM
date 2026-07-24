#! /bin/bash
# Stage 1: Confidence-based Explicit Meme Identification (pseudo-label the explicit set)
python src/main.py --config-name label_qwen_MAMI
# Stage 2: Multimodal Retrieval-based Contrastive Pairing (embeddings + contrastive pairs)
python src/model/Experience/make_embeddings.py --dataset MAMI
python src/model/Experience/conduct_retrieval.py --dataset MAMI
# Stage 3: Pairwise Learning-guided Experience Gathering
python src/main.py --config-name experience_qwen_MAMI
# Stage 4: Reference Refinement
python src/main.py --config-name reference_qwen_MAMI
# Stage 5: Inference (predict the subtle set with the self-generated references)
python src/main.py --config-name inpredict_qwen_MAMI
