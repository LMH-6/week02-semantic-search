import streamlit as st

from compare_search import semantic_search
from search_keyword import keyword_search
from faq_data import faqs

# 语义检索最低相似度阈值（初步实验值）
SIMILARITY_THRESHOLD = 0.2465

# 页面基本配置
st.set_page_config(
    page_title="Local Semantic Search",
    page_icon="🔍",
    layout="wide"
)

# 页面标题
st.title("Local Knowledge Search")
st.caption("A lightweight FAQ retrieval system")

# 1. 用户输入
query = st.text_input(
    "Enter your question",
    placeholder="e.g. What is an AI agent?"
)

# 2. 选择检索算法
method = st.radio(
    "Retrieval Method",
    ["Semantic Search", "Keyword Search"],
    horizontal=True
)

# 3. 设置 Top-K
top_k = st.slider(
    "Number of results (Top-K)",
    min_value=1,
    max_value=10,
    value=5
)


# 4. 搜索按钮
if st.button("Search", type="primary"):

    if not query.strip():
        st.warning("Please enter a question.")

    else:
        # 根据用户选择执行不同的检索算法
        if method == "Semantic Search":

            results = semantic_search(query, top_k)

            if results:
                # 获取 Top-1 的相似度
                max_score = results[0]["score"]

                if max_score < SIMILARITY_THRESHOLD:
                    st.warning(
                        f"No sufficiently similar candidate found. "
                        f"Top-1 Score: {max_score:.4f}"
                    )
                    results = []

                else:
                    st.info(
                        "Similar candidate materials found. "
                        "Please check their relevance."
                    )

            else:
                st.warning("No candidate results returned.")

        else:
            # 关键词检索
            results = keyword_search(query, top_k)

            # 只保留真正匹配到关键词的结果
            results = [
                item for item in results
                if item["score"] > 0
            ]

            if not results:
                st.warning("No matching keywords found.")

        # 5. 展示检索结果
        if results:

            st.subheader("Search Results")

            faq_lookup = {
                faq["id"]: faq for faq in faqs
            }

            for rank, item in enumerate(results, start=1):

                faq = faq_lookup[item["faq_id"]]

                with st.container(border=True):

                    st.markdown(
                        f"**{rank}. {item['faq_id']}**"
                    )

                    st.write(faq["question"])
                    st.write(faq["answer"])

                    if method == "Semantic Search":

                        st.metric(
                            "Cosine Similarity",
                            f"{item['score']:.4f}"
                        )

                    else:

                        st.metric(
                            "Keyword Score",
                            item["score"]
                        )

                        st.caption(
                            f"Matched keywords: {item['matched_tokens']}"
                        )
