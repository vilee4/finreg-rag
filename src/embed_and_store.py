import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download   # 使用modelscope 国内镜像加速模型下载
from chromadb import errors

def getFullPath(*paths):
    """获取项目内的完整路径"""
    projetc_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(projetc_path)
    return os.path.join(project_root,*paths)

def main():
    # 1.加载文本块数据
    chunk_data = getFullPath("data","chunks.json")
    with open(chunk_data,"r",encoding="utf-8") as f:
        chunks = json.load(f)
    texts = [c['text'] for c in chunks]
    ids = [str(c['id']) for c in chunks]
    print(f'📄 加载{len(texts)}个文本块')

    # 2. 加载BGE中文Embedding模型
    print("⏳ 正在加载BGE模型...")
    # # 加载预训练模型
    # model=SentenceTransformer(
    #     'BAAI/bge-large-zh-v1.5',
    #     trust_remote_code=True  ##  允许加载自定义模型代码
    # ) 
    # model.max_seq_length = 512 # 提升长文本处理能力

    # 国内镜像下载模型 
    # 下载到本地
    model_dir = snapshot_download(
        'BAAI/bge-large-zh-v1.5',
        cache_dir="D:/aiCache"    
    )
    print(f"模型下载路径{model_dir}")
    model = SentenceTransformer(model_dir)
    model.max_seq_length = 512

    # 3. 生成embedding
    print("正在生成embding...")
    embeddings = model.encode(texts,normalize_embeddings=True, show_progress_bar= True)
    print(f"生成{len(embeddings)}个向量，维度：{len(embeddings[0])}")

    # 4. 初始化chroma并存入
    vector_db_path = getFullPath("data","vector_db")
    client = chromadb.PersistentClient(path=vector_db_path)

    # 删除已存在的集合，避免重复
    collection_name = "finreg_chunks"
    try:
        client.delete_collection("finreg_chunks")
    except errors.NotFoundError:
        pass # 如果不存在则直接跳过

    # 创建新集合  chroma 会自动存储原始数据 + 向量
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} # 余弦相似度
    )

    # 批量添加
    collection.add(
        ids = ids,
        documents =  texts,
        embeddings = embeddings.tolist() # 将list转为json 嵌入向量函数 生成文本的向量
    )
    print(f"向量数据已保存至：{vector_db_path}")

    # 简单检索测试
    print("\n 测试检索 east客户id字段的要求")
    query = 'EAST客户id字段要求'
    query_embedding = model.encode(
        query
        ,normalize_embeddings=True # 确保余弦相似度计算是准确的
     ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    for i,doc in enumerate(results["documents"][0]):
        print(f"\n 结果{i+1}:")
        print(doc[:200] + "..." if len(doc)>200 else doc)

if __name__ == "__main__":
    main()