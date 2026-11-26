import re

from faq_data import faqs
from test_queries import test_queries

STOP_WORDS = {
    "what", "is", "in", "how", "does",
    "a", "an", "the", "to", "and",
    "of", "for", "can", "do", "i",
    "which", "be", "on", "my", "it",
    "that", "this", "are", "with"
}

def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    # 接下来需要使用 set() 和 STOP_WORDS
    words = set(words) - STOP_WORDS
    return words

def keyword_search(query, top_k=3):

    # 1. 提取用户查询的关键词
    query_tokens = tokenize(query)

    # 2. 创建一个列表，保存所有 FAQ 的匹配结果
    results = []

    # 3. 遍历全部 FAQ
    for faq in faqs:
        # 提取当前 FAQ 的关键词
        faq_tokens = tokenize(faq["question"])
        # 计算查询和 FAQ 的共同关键词
        matched_tokens = query_tokens & faq_tokens
        # 计算共同关键词的数量
        score = len(matched_tokens)
        # 将结果保存到 results 中
        results.append({
            "faq_id": faq["id"],
            "question": faq["question"],
            "score": score,
            "matched_tokens": list(matched_tokens)
        })
    # 4. 按照匹配分数从高到低排序
    results.sort(key=lambda x: x["score"], reverse=True)

    # 5. 返回前 top_k 条结果
    return results[:top_k]

if __name__ == "__main__":

    # 批量测试关键词检索

    hit_count = 0
    answerable_count = 0

    for test in test_queries:

        print("\n" + "=" * 60)
        print(f"Test ID: {test['id']}")
        print(f"Type: {test['type']}")
        print("Query:", test["query"])

        # 执行关键词检索
        results = keyword_search(test["query"], top_k=3)

        print("\nKeyword Top-3:")

        for rank, item in enumerate(results, start=1):
            print(f"{rank}. ID: {item['faq_id']}")
            print(f"   Score: {item['score']}")
            print(f"   Matched: {item['matched_tokens']}")
            print(f"   Question: {item['question']}")

        # 有答案的查询：计算 Hit@3
        if test["relevant_ids"]:
            answerable_count += 1

            hit = any(
                item["faq_id"] in test["relevant_ids"]
                for item in results
            )

            hit_count += int(hit)

            print("Hit@3:", hit)

        # 无答案查询：单独记录
        else:
            max_score = results[0]["score"] if results else 0
            print("No-answer test")
            print("Max keyword score:", max_score)


    # 汇总有答案查询的 Hit@3
    print("\n" + "=" * 60)
    print("Keyword Search Evaluation")
    print(f"Hit@3: {hit_count}/{answerable_count}")
    print(f"Hit@3 Rate: {hit_count / answerable_count:.2%}")
