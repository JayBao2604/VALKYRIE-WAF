# pip install llama-index-llms-gemini llama-index-embeddings-huggingface llama-index beautifulsoup4
import os
import json
import argparse
from pathlib import Path
from bs4 import BeautifulSoup
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Cấu hình Gemini API
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyCOup9Sbs_o3MbZBR5i7yvKRac_fvFVn9U")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Cấu hình LLM và Embedding model
Settings.llm = Gemini(model="models/gemini-flash-latest", api_key=GEMINI_API_KEY)
# Sử dụng embedding local (miễn phí, không cần API)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_bugcrowd_html(file_path: str) -> list[Document]:
    """Parse Bugcrowd HTML file (chứa các <article> tags)"""
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()
    
    soup = BeautifulSoup(html, "html.parser")
    docs = []
    
    # Tìm tất cả các <article> tags
    articles = soup.find_all("article")
    
    if not articles:
        print(f"⚠️  No <article> tags found in {file_path}")
        return []
    
    # Parse từng article
    for idx, article in enumerate(articles, 1):
        text = article.get_text("\n", strip=True)
        if text:
            # Xác định section type
            section_type = "summary"
            if article.find("code") or article.find("pre"):
                section_type = "code"
            elif "Activity" in text:
                section_type = "activity"
            
            docs.append(Document(
                text=text,
                metadata={
                    "source": file_path,
                    "platform": "bugcrowd",
                    "article_index": idx,
                    "section_type": section_type
                }
            ))
    
    return docs


def parse_hackerone_json(file_path: str) -> list[Document]:
    """Parse HackerOne JSON file (chỉ lấy trường vulnerability_information)"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Lấy vulnerability_information
    vuln_info = data.get("vulnerability_information", "")
    
    if not vuln_info:
        print(f"⚠️  No 'vulnerability_information' field found in {file_path}")
        return []
    
    # Tạo document từ vulnerability_information
    docs = [
        Document(
            text=vuln_info,
            metadata={
                "source": file_path,
                "platform": "hackerone",
                "report_id": data.get("id"),
                "title": data.get("title"),
                "severity": data.get("severity_rating"),
                "cve_ids": data.get("cve_ids", [])
            }
        )
    ]
    
    return docs


def parse_report(file_path: str) -> list[Document]:
    """Auto-detect platform và parse report tương ứng"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Detect platform từ extension hoặc path
    if file_path.suffix.lower() == ".json":
        print(f"📄 Parsing HackerOne JSON: {file_path.name}")
        return parse_hackerone_json(str(file_path))
    elif file_path.suffix.lower() in [".html", ".htm"]:
        print(f"📄 Parsing Bugcrowd HTML: {file_path.name}")
        return parse_bugcrowd_html(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


# ============================================================================
# RAG FUNCTIONS
# ============================================================================

def create_rag_index(docs: list[Document]) -> VectorStoreIndex:
    """Tạo RAG index từ documents"""
    if not docs:
        raise ValueError("No documents to index")
    
    print(f"📚 Creating index from {len(docs)} documents...")
    index = VectorStoreIndex.from_documents(docs)
    return index


def query_report(index: VectorStoreIndex, query: str, verbose: bool = False) -> str:
    """Query RAG index"""
    query_engine = index.as_query_engine(
        similarity_top_k=3,  # Retrieve top-3 most relevant chunks
        response_mode="compact"
    )
    response = query_engine.query(query)
    
    # Show retrieved chunks if verbose
    if verbose and hasattr(response, 'source_nodes'):
        print(f"\n{'='*80}")
        print(f"🔍 RAG RETRIEVAL DEBUG")
        print(f"{'='*80}")
        print(f"Retrieved {len(response.source_nodes)} relevant chunks:\n")
        
        for i, node in enumerate(response.source_nodes, 1):
            score = node.score if hasattr(node, 'score') else 'N/A'
            text_preview = node.text[:200] + "..." if len(node.text) > 200 else node.text
            print(f"Chunk {i} (Relevance: {score}):")
            print(f"  {text_preview}\n")
        
        print(f"{'='*80}\n")
    
    return str(response)


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG Analyzer for Disclosure Reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  # Analyze Bugcrowd HTML report
  python rag_analyzer.py data/reports/bugcrowd/cross-site-scripting-xss.html
  
  # Analyze HackerOne JSON report
  python rag_analyzer.py data/reports/hackerone/hackerone_3404968.json
  
  # Custom query
  python rag_analyzer.py report.html --query "What attack techniques were used?"
        """
    )
    
    parser.add_argument(
        "report_file",
        type=str,
        help="path to report file (HTML for Bugcrowd, JSON for HackerOne)"
    )
    
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="Summarize the vulnerability report, including attack techniques, payloads used, and impact.",
        help="question to ask about the report"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="show RAG retrieval debug information (retrieved chunks and scores)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"  RAG Analyzer - Disclosure Reports")
    print(f"{'='*80}\n")
    
    # Parse report
    try:
        docs = parse_report(args.report_file)
        print(f"✅ Parsed {len(docs)} documents\n")
        
        # Create index
        index = create_rag_index(docs)
        print(f"✅ Index created\n")
        
        # Query
        print(f"🔍 Query: {args.query}\n")
        
        if args.verbose:
            print(f"🧠 Embedding Model: {Settings.embed_model.__class__.__name__}")
            print(f"🤖 LLM: {Settings.llm.__class__.__name__}\n")
        
        print(f"{'-'*80}")
        response = query_report(index, args.query, verbose=args.verbose)
        print(response)
        print(f"{'-'*80}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


