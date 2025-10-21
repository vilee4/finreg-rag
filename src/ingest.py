import os
os.environ["UNSTRUCTURED_SKIP_NLTK"] = "1"  # ← 关键！

import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text
from typing import List, Dict, Any
from datetime import datetime
from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download  # 使用modelscope 国内镜像加速模型下载

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
print(PROJECT_ROOT)
# 加载配置文件
load_dotenv()


def getFullPath(*paths):
    """获取项目内的完整路径"""
    projetc_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(projetc_path)
    return os.path.join(project_root, *paths)


# 初始化 chroma
vector_db_path = getFullPath("data", "vector_db")
chromadb_client = chromadb.PersistentClient(vector_db_path)
collection_name = "finreg_chunks"


# 获取文件哈希值判断文件是否已经存在
def get_file_hash(file_path: Path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:  # 二进制读
        for chunk in iter(lambda: f.read(4096), b""):  # 返回空时结束
            hasher.update(chunk)
    return hasher.hexdigest()


def get_exists_file_hashes() -> dict[str, str]:
    chunks_file = Path(getFullPath("data", "chunks.json"))
    """从 chunks.json 读取已处理文件的 hash 记录"""
    if not chunks_file.exists():
        return {}
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        # 提取所有元数据中的 file_hash
        return {
            c["metadata"]["file_hash"]: c["metadata"]["file_path"]
            for c in chunks
            if "metadata" in c
        }


def parse_document(file_path: Path) -> list[str]:
    # 根据文件后缀解析文本
    suffix = file_path.suffix.lower()
    elements = []

    try:
        if ".pdf" == suffix:
            elements = partition_pdf(
                filename=str(file_path), strategy="fast", infer_table_structure=False
            )
        elif suffix == ".docx":
            elements = partition_docx(filename=str(file_path))
        elif suffix == ".txt":
            # 不使用understruct 解析 txt
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if not content.strip():
                return []
            
            # 按双换行（段落）分割
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # 如果段落太少（可能无空行分隔），尝试按单行智能合并
            if len(paragraphs) <= 1:
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                paragraphs = []
                current = ""
                for line in lines:
                    # 启发式：短行（<50字符）很可能是上一句的延续
                    if current and (len(line) < 50 or line.endswith(('，', '、', '；'))):
                        current += " " + line
                    else:
                        if current:
                            paragraphs.append(current)
                        current = line
                if current:
                    paragraphs.append(current)

            # 过滤掉过短的段落
            texts = [p for p in paragraphs if len(p) > 20]
            return texts
        else:
            print(f"不支持的文件格式：{suffix}")
            return []
    except Exception as e:
        print(f"解析{file_path.name}失败：{e}")
        return []

    # 提取文本 （过滤标题、页眉等）
    texts = []
    for el in elements:
        if el.category in ["NarrativeText", "Title", "ListItem"]:
            text = str(el).strip()
            if text and len(text) > 20:  # 过滤短文本
                texts.append(text)
    return texts


# 简单按段落分块
def chunk_text(
    texts: List[str], file_path: Path, file_hash: str, max_chunk_size: int = 512
) -> List[Dict[str, Any]]:
    chunks = []
    current_chunk = ""
    chunk_id_base = f"{file_path.stem}_{file_hash[:8]}"  # 文件名——hash

    for i, para in enumerate(texts):
        if len(current_chunk) + len(para) < max_chunk_size:
            current_chunk += "\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(
                    {
                        "id": f"{chunk_id_base}_{len(chunks)}",
                        "text": current_chunk,
                        "metadata": {
                            "file_path": str(file_path.relative_to(PROJECT_ROOT)),
                            "file_hash": file_hash,
                            "chunk_index": len(chunks),
                            "ingest_time": datetime.now().isoformat(),
                        },
                    }
                )
            current_chunk = para

    # 添加最后一块
    if current_chunk:
        chunks.append(
            {
                "id": f"{chunk_id_base}_{len(chunks)}",
                "text": current_chunk,
                "metadata": {
                    "file_path": str(file_path.relative_to(PROJECT_ROOT)),
                    "file_hash": file_hash,
                    "chunk_index": len(chunks),
                    "ingest_time": datetime.now().isoformat(),
                },
            }
        )
    return chunks


# 增量更新向量库
def update_vector_db(new_chunks: List[Dict]):
    if not new_chunks:
        return

    # 加载 Embedding 模型
    model_dir = snapshot_download("BAAI/bge-large-zh-v1.5", cache_dir="D:/aiCache")
    print(f"模型下载路径{model_dir}")
    model = SentenceTransformer(model_dir)
    model.max_seq_length = 512

    # 生成向量
    texts = [c["text"] for c in new_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    ids = [c["id"] for c in new_chunks]
    metadatas = [c["metadata"] for c in new_chunks]

    # 获取或创建集合
    try:
        collection = chromadb_client.get_collection(collection_name)
    except:
        collection = chromadb_client.create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )


def ingest_file(file_path: Path, existing_hashes: Dict[str, str]):
    """处理单个文件"""
    if not file_path.exists():
        print(f" 文件不存在: {file_path}")
        return []

    file_hash = get_file_hash(file_path)
    if file_hash in existing_hashes:
        print(f" 跳过已处理文件: {file_path.name}")
        return []

    print(f" 正在解析: {file_path.name}")
    texts = parse_document(file_path)
    if not texts:
        print(f"  未提取到有效文本: {file_path.name}")
        return []

    chunks = chunk_text(texts, file_path, file_hash)
    print(f"  → 生成 {len(chunks)} 个文本块")
    return chunks


def save_chunks_to_json(all_chunks: List[Dict]):
    """保存所有 chunks 到 JSON（用于备份和调试）"""
    # 读取旧 chunks
    old_chunks = []
    chunks_file = Path(getFullPath("data", "chunks.json"))
    if chunks_file.exists():
        with open(chunks_file, "r", encoding="utf-8") as f:
            old_chunks = json.load(f)

    # 合并（按 id 去重）
    chunk_dict = {c["id"]: c for c in old_chunks}
    for c in all_chunks:
        chunk_dict[c["id"]] = c

    # 写回
    chunks_file.parent.mkdir(exist_ok=True)
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(list(chunk_dict.values()), f, ensure_ascii=False, indent=2)
    print(f" 已保存 {len(chunk_dict)} 个片段到 {chunks_file}")


# === 主程序 ===
def main():
    import argparse

    parser = argparse.ArgumentParser(description="金融法规文档摄入工具")
    parser.add_argument("--file", type=str, help="单个文件路径")
    parser.add_argument("--dir", type=str, help="文件夹路径")
    args = parser.parse_args()

    if not args.file and not args.dir:
        parser.error("请指定 --file 或 --dir")

    # 获取已处理文件 hash
    existing_hashes = get_exists_file_hashes()
    all_new_chunks = []

    if args.file:
        file_path = Path(args.file).resolve() # 转绝对路径
        all_new_chunks.extend(ingest_file(file_path, existing_hashes))
    elif args.dir:
        dir_path = Path(args.dir).resolve()
        for ext in ["*.pdf", "*.docx", "*.txt"]:
            for file_path in dir_path.glob(ext):
                all_new_chunks.extend(ingest_file(file_path, existing_hashes))

    if all_new_chunks:
        print(f"\n共新增 {len(all_new_chunks)} 个片段")
        update_vector_db(all_new_chunks)
        save_chunks_to_json(all_new_chunks)
    else:
        print("无新文件需要处理")


if __name__ == "__main__":
    main()
