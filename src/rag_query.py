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
        include=["documents", "distances"],
    )

    # 拼接上下文
    contents = results["documents"][0]
    context_text = "\n\n".join([f"片段{i+1}:\n{ctx}" for i, ctx in enumerate(contents)])

    print(f"检索到{len(contents)}个相关规则片段")
    for i, ctx in enumerate(contents):
        print(f"----片段{i+1}----\n{ctx[:200]}...")

    # 构建 prompt 金融合规场景
    prompt = f"""
    你是一名专业的金融合规顾问，请根据以下监管规定片段，严谨、准确地回答用户问题。
    要求：
    1. 仅基于提供的片段回答，不要编造
    2. 引用具体条款（如“根据片段1”）
    3. 若片段不相关或信息不足，请回答“根据现有资料无法确定”

    监管规定片段：
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
    )

    if response.status_code == 200:
        return response.output.text
    else:
        raise Exception(f"Qwen 调用失败: {response.code} - {response.message}")


# 主程序
if __name__ == "__main__":
    question = "EAST报送中客户ID字段的要求是什么？"
    try:
        answer = rag_query(question)
        print("\n✅ 最终回答：")
        print("=" * 60)
        print(answer)
        print("=" * 60)
    except Exception as e:
        print(f"错误：{e}")
