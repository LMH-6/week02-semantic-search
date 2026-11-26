import torch
from sentence_transformers import SentenceTransformer

from faq_data import faqs
from test_queries import test_queries
from search_keyword import keyword_search

# 1. 加载本地 FAQ 索引
index = torch.load(
    "faq_index.pt",
    map_location="cpu",
    weights_only=True
)

index_faqs = index["faqs"]
corpus_embeddings = index["corpus_embeddings"]

# 2. 检查关键词检索和语义检索的 FAQ 顺序是否一致
if [faq["id"] for faq in faqs] != [faq["id"] for faq in index_faqs]:
    raise ValueError("FAQ 数据与向量索引不一致，请重新构建索引")

# 3. 加载模型
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device="cpu")

def semantic_search(query, top_k=3):

    # 1. 编码用户查询
    query_embedding = embedder.encode_query(query,convert_to_tensor=True)

    # 2. 计算相似度
    similarity_scores = embedder.similarity(query_embedding,corpus_embeddings)[0]

    # 3. 获取 Top-K
    k = min(top_k, len(index_faqs))
    scores, indices = torch.topk(similarity_scores, k=k)

    # 4. 整理检索结果
    results = []

    for score, idx in zip(scores, indices):

        faq = index_faqs[idx.item()]

        results.append({
            "faq_id": faq["id"],
            "question": faq["question"],
            "score": score.item()
        })

    return results

if __name__ == "__main__":

    # 初始化评价计数器
    keyword_hits = 0
    semantic_hits = 0
    answerable_count = 0

    positive_top_scores = []
    negative_top_scores = []

    # 遍历全部测试查询
    for test in test_queries:

        query = test["query"]
        relevant_ids = test["relevant_ids"]

        print("\n" + "=" * 65)
        print(f"Test ID: {test['id']}")
        print("Query:", query)

        # 1. 执行关键词检索
        keyword_results = [item for item in keyword_search(query, top_k=3) if item["score"] > 0]

        # 2. 执行语义检索
        semantic_results = semantic_search(query, top_k=3)

        semantic_max = (semantic_results[0]["score"] if semantic_results else None)

        # 3. 输出关键词检索结果
        print("\n[Keyword Search]")

        for rank, item in enumerate(keyword_results, start=1):
            print(
                f"{rank}. {item['faq_id']} | "
                f"Score: {item['score']} | "
                f"Matched: {item['matched_tokens']}"
            )

        # 4. 输出语义检索结果
        print("\n[Semantic Search]")

        for rank, item in enumerate(semantic_results, start=1):
            print(
                f"{rank}. {item['faq_id']} | "
                f"Score: {item['score']:.4f}"
            )

        # 5. 有答案的查询：计算 Hit@3
        if relevant_ids:

            if semantic_max is not None:
                positive_top_scores.append(semantic_max)

            answerable_count += 1
            keyword_hit = any(item["faq_id"] in relevant_ids for item in keyword_results)
            semantic_hit = any(item["faq_id"] in relevant_ids for item in semantic_results)

            keyword_hits += int(keyword_hit)
            semantic_hits += int(semantic_hit)

            print("\n[Evaluation]")
            print("Relevant IDs:", relevant_ids)
            print("Keyword Hit@3:", keyword_hit)
            print("Semantic Hit@3:", semantic_hit)

        # 6. 无答案查询：单独记录最高分
        else:

            if semantic_max is not None:
                negative_top_scores.append(semantic_max)

            keyword_max = (keyword_results[0]["score"] if keyword_results else 0)
            semantic_max = (semantic_results[0]["score"] if semantic_results else 0)

            print("\n[No-answer Evaluation]")
            print("Keyword Max Score:", keyword_max)
            print(f"Semantic Max Score: {semantic_max:.4f}")

    # 7. 汇总两种算法的评价结果
    print("\n" + "=" * 65)
    print("FINAL COMPARISON")

    print(
        f"Keyword Hit@3: "
        f"{keyword_hits}/{answerable_count} "
        f"({keyword_hits / answerable_count:.2%})"
    )

    print(
        f"Semantic Hit@3: "
        f"{semantic_hits}/{answerable_count} "
        f"({semantic_hits / answerable_count:.2%})"
    )

    print("\nSemantic Score Analysis")

    if positive_top_scores:
        print(
            "Answerable Query Score Range:",
            f"{min(positive_top_scores):.4f}",
            "-",
            f"{max(positive_top_scores):.4f}"
        )

    if negative_top_scores:
        print(
            "No-answer Query Score Range:",
            f"{min(negative_top_scores):.4f}",
            "-",
            f"{max(negative_top_scores):.4f}"
        )

    if positive_top_scores and negative_top_scores:

        min_positive = min(positive_top_scores)
        max_negative = max(negative_top_scores)

        if min_positive > max_negative:

            threshold = (min_positive + max_negative) / 2

            print(f"Provisional Threshold: {threshold:.4f}")

        else:
            print("Warning: Score ranges overlap.")
            print("A single threshold cannot separate all tests.")