import torch
from sentence_transformers import SentenceTransformer
from faq_data import faqs

corpus=[faq["question"] for faq in faqs]
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device="cpu")
corpus_embeddings = embedder.encode_document(corpus,convert_to_tensor=True)
torch.save(
    {
        "faqs": faqs,
        "corpus_embeddings": corpus_embeddings
    },
    "faq_index.pt"
)
print("FAQ 数量:", len(faqs))
print("向量矩阵形状:", corpus_embeddings.shape)
print("索引保存成功: faq_index.pt")