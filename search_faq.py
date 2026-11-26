import torch
from sentence_transformers import SentenceTransformer
from test_queries import test_queries

# 加载本地索引
index = torch.load("faq_index.pt",map_location="cpu",weights_only=True)

faqs = index["faqs"]
corpus_embeddings = index["corpus_embeddings"]

print("FAQ 数量:", len(faqs))
print("向量矩阵形状:", corpus_embeddings.shape)

# 加载模型，用于编码用户查询
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device="cpu")

# 批量测试语义检索

hit_count = 0
answerable_count = 0

for test in test_queries:

    query = test["query"]

    # 1. 将用户查询编码成向量
    query_embedding = embedder.encode_query(query,convert_to_tensor=True)

    # 2. 计算查询与全部 FAQ 的相似度
    similarity_scores = embedder.similarity(query_embedding,corpus_embeddings)[0]

    # 3. 获取 Top-3
    top_k = min(3, len(faqs))
    scores, indices = torch.topk(similarity_scores, k=top_k)

    retrieved_ids = []

    print("\n" + "=" * 60)
    print("Test ID:", test["id"])
    print("Query:", query)
    print("Semantic Top-3:")

    # 4. 输出检索结果
    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        faq = faqs[idx.item()]
        retrieved_ids.append(faq["id"])

        print(f"\n{rank}. ID: {faq['id']}")
        print("Question:", faq["question"])
        print("Answer:", faq["answer"])
        print(f"Score: {score.item():.4f}")

    # 5. 计算有答案查询的 Hit@3
    if test["relevant_ids"]:
        answerable_count += 1
        hit = any(doc_id in test["relevant_ids"] for doc_id in retrieved_ids)
        hit_count += int(hit)
        print("\nHit@3:", hit)

    # 6. 无答案查询单独记录
    else:
        max_score = scores[0].item() if len(scores) > 0 else 0

        print("\nNo-answer test")
        print(f"Max semantic similarity: {max_score:.4f}")


# 7. 汇总实验结果
print("\n" + "=" * 60)
print("Semantic Search Evaluation")
print(f"Hit@3: {hit_count}/{answerable_count}")
print(f"Hit@3 Rate: {hit_count / answerable_count:.2%}")