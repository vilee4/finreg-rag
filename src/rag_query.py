import os
import dashscope
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download
import chromadb
from dashscope import Generation

# 加载环境变量，获取 sdk
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
if not dashscope.api_key:
    raise ValueError("请配置 api key")


def getFullPath(*paths):
    """获取项目内的完整路径"""
    projetc_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(projetc_path)
    return os.path.join(project_root, *paths)


# 初始化组件
print("正在加载 Embedding 模型...")
model_dir = snapshot_download("BAAI/bge-large-zh-v1.5", cache_dir="D:/aiCache")
embed_model = SentenceTransformer(model_dir)

vector_db_path = getFullPath("data", "vector_db")
client = chromadb.PersistentClient(vector_db_path)
collection = client.get_collection("finreg_chunks")


# rag查询函数
def rag_query(user_question: str, top_k: int = 3) -> str:
    # 生成查询向量
    query_embedding = embed_model.encode(
        user_question, normalize_embeddings=True
    ).tolist()

    # 检索相关片段
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "distances"],  ## Chroma（>=0.4.22） 中include 不在支持 ids 显示指定，总是返回
    )

    # 构建 [片段1]: ... 格式
    context_blocks = []
    source = []
    for i, (doc,doc_id, dist) in enumerate(zip(
        results["documents"][0]
        ,results["ids"][0]
        ,results["distances"][0])
    ):
        similarity = 1-dist   #  转为相似度 (0-1)
        snippet = doc[:200].replace("\n", " ") + ("..." if len(doc) > 200 else "")

        context_blocks.append(f"[片段{i+1}]: {doc}")
        source.append({
            "id": doc_id,
            "text": doc,
            "similarity": round(similarity,4),  # 距离转相似度
            "snippet": snippet
        })

    # # 拼接上下文
    # contents = results["documents"][0]
    context_text = "\n\n".join(context_blocks)

    # 构建 prompt 金融合规场景
    prompt = f"""
    你是一名专业的金融合规顾问，请严格根据以下【监管原文片段】回答用户问题。

    回答规则：
    1. 仅使用【监管原文片段】中的信息，禁止编造、推测或使用外部知识。
    2. 若问题无法从片段中回答，请明确说：“根据当前提供的监管资料，无法回答该问题。”
    3. 引用时必须使用格式：[片段1]、[片段2] 等。
    4. 回答应简洁、专业、准确。

    【监管原文片段】
    {context_text}

    用户问题：{user_question}

    请回答：
    """

    # 调用qwen
    print("\n 正在调用qwen大模型...")
    response = Generation.call(
        model="qwen-turbo",
        prompt=prompt,
        temperature=0.3,  ## 降低随机性，提高准确性
        max_tokens=500,
        top_p=0.8
    )

    if response.status_code == 200:
        answer = response.output.text.strip()
        return {
            "answer": answer,
            "source": source
        }
    else:
        raise Exception(f"Qwen 调用失败: {response.code} - {response.message}")


# 主程序
if __name__ == "__main__":
    print("="*60)
    print("金融合规RAG，带溯源")
    print("输入问题，将返回答案和依据")
    print("输入 'quit' 或 'exit' 退出系统")
    print("="*60)
    
    while True:
        try:
            question = input("\n 请输入您的问题：").strip()
            if not question:
                continue
            if question.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            result = rag_query(question,top_k=3)
            # 输出答案
            print(f"\n回答:\n{result['answer']}")
            
            # 输出溯源
            print(f"\n依据来源（按相关性排序）:")
            for i, src in enumerate(result['source'], 1):
                print(f"\n--- [片段{i}] (相似度: {src['similarity']:.4f}) ---")
                print(f"原始ID: {src['id']}")
                print(f"内容: {src['snippet']}")
        except KeyboardInterrupt:
            print("中途退出！")
            break
        except Exception as e:
            print(f" 错误: {e}")