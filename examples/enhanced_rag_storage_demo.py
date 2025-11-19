#!/usr/bin/env python3
# generated: python -m src.cli metadata-injector @ 2025-11-19T13:04:42Z
# classification: developer
"""
增强版 RAG 和存储适配演示脚本

展示基于 LlamaIndex 最佳实践的增强版组件功能：
- EnhancedVectorStoreManager：LlamaIndex Chroma 集成
- EnhancedGraphStoreManager：多后端图存储支持
- EnhancedIndexService：AutoRetriever 智能检索
- 性能监控和指标收集
"""

import os
import time
import logging
from typing import List, Dict, Any

from llama_index.core.schema import Document as LlamaDocument

from src.framework.shared.logging import get_logger
from src.framework.storage.enhanced_vector_store import EnhancedVectorStoreManager
from src.framework.storage.enhanced_graph_store import EnhancedGraphStoreManager
from src.framework.rag.enhanced_index_service import EnhancedIndexService
from src.framework.shared.models import DocumentChunk

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def create_sample_documents() -> List[LlamaDocument]:
    """创建示例文档"""
    documents = [
        LlamaDocument(
            text="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。"
                 "Python 以其简洁易读的语法而闻名，支持多种编程范式，包括面向对象、"
                 "函数式和过程式编程。Python 拥有庞大的标准库和活跃的社区。",
            metadata={"source": "python_intro", "category": "programming", "author": "Guido"}
        ),
        LlamaDocument(
            text="机器学习是人工智能的一个分支，专注于开发算法和统计模型，"
                 "使计算机系统能够执行特定任务而无需使用明确的指令。"
                 "机器学习算法基于样本数据构建预测或决策模型。",
            metadata={"source": "ml_intro", "category": "ai", "author": "ML Expert"}
        ),
        LlamaDocument(
            text="深度学习是机器学习的一个子集，使用包含多个隐藏层的神经网络。"
                 "深度学习在图像识别、自然语言处理和语音识别等领域取得了突破性进展。",
            metadata={"source": "deep_learning", "category": "ai", "author": "DL Researcher"}
        ),
        LlamaDocument(
            text="ChromaDB 是一个开源的向量数据库，专为 AI 应用程序设计。"
                 "它提供简单的 API 来存储和查询嵌入向量，支持持久化、过滤和混合搜索。",
            metadata={"source": "chromadb", "category": "database", "author": "Chroma Team"}
        ),
        LlamaDocument(
            text="LlamaIndex 是一个数据框架，用于构建 LLM 应用程序。"
                 "它提供了连接、索引和访问私有或特定领域数据的工具和抽象。"
                 "LlamaIndex 支持多种数据源和向量存储。",
            metadata={"source": "llamaindex", "category": "framework", "author": "LlamaIndex Team"}
        ),
        LlamaDocument(
            text="向量嵌入是将文本、图像或其他数据转换为数值向量的过程。"
                 "这些向量捕获数据的语义特征，使得相似的内容在向量空间中距离更近。"
                 "嵌入广泛用于搜索、推荐和聚类任务。",
            metadata={"source": "embeddings", "category": "ml_concept", "author": "ML Professor"}
        ),
        LlamaDocument(
            text="图数据库使用图结构存储数据，其中节点表示实体，"
                 "边表示实体之间的关系。Neo4j 是最流行的图数据库之一，"
                 "适用于社交网络分析、推荐系统和欺诈检测。",
            metadata={"source": "graph_db", "category": "database", "author": "Graph Expert"}
        ),
        LlamaDocument(
            text="自然语言处理（NLP）是人工智能的一个领域，"
                 "专注于计算机与人类语言之间的交互。"
                 "NLP 技术包括文本分类、情感分析、机器翻译和问答系统。",
            metadata={"source": "nlp", "category": "ai", "author": "NLP Specialist"}
        ),
        LlamaDocument(
            text="RAG（检索增强生成）是一种 AI 架构，"
                 "结合了信息检索和文本生成。RAG 系统首先检索相关文档，"
                 "然后使用这些文档作为上下文来生成更准确和相关的响应。",
            metadata={"source": "rag", "category": "ai", "author": "RAG Researcher"}
        ),
        LlamaDocument(
            text="知识图谱是结构化知识的图形表示，"
                 "其中节点表示概念或实体，边表示它们之间的关系。"
                 "知识图谱广泛用于搜索引擎、推荐系统和语义搜索。",
            metadata={"source": "knowledge_graph", "category": "ai", "author": "Knowledge Engineer"}
        )
    ]
    return documents


def demonstrate_enhanced_vector_store():
    """演示增强版向量存储管理器"""
    print("\n" + "="*60)
    print("🔍 演示增强版向量存储管理器")
    print("="*60)
    
    try:
        # 初始化增强版向量存储管理器
        vector_manager = EnhancedVectorStoreManager(persist_dir="./vector/enhanced_chroma")
        
        # 创建示例文档
        documents = create_sample_documents()
        print(f"✅ 创建了 {len(documents)} 个示例文档")
        
        # 创建索引
        print("🚀 创建向量索引...")
        index = vector_manager.create_index(documents, collection_name="demo_docs")
        print("✅ 索引创建成功")
        
        # 获取集合信息
        collection_info = vector_manager.get_collection_info("demo_docs")
        print(f"📊 集合信息: {collection_info}")
        
        # 创建查询引擎
        print("🎯 创建查询引擎...")
        query_engine = vector_manager.create_query_engine("demo_docs", similarity_top_k=3)
        
        # 执行查询
        test_queries = [
            "什么是 Python 编程语言？",
            "机器学习和深度学习有什么区别？",
            "向量数据库有哪些？",
            "什么是 RAG 架构？"
        ]
        
        for query in test_queries:
            print(f"\n❓ 查询: {query}")
            start_time = time.time()
            response = query_engine.query(query)
            query_time = time.time() - start_time
            print(f"⏱️  查询耗时: {query_time:.3f}s")
            print(f"📄 响应: {str(response)[:200]}...")
        
        # 演示高级功能
        print("\n🔧 演示高级功能...")
        
        # 添加新文档
        new_doc = LlamaDocument(
            text="AutoRetriever 是 LlamaIndex 的智能检索器，"
                 "能够自动选择最佳的检索策略。",
            metadata={"source": "autoretriever", "category": "llamaindex"}
        )
        
        vector_manager.add_documents_to_index([new_doc], "demo_docs")
        print("✅ 添加新文档到索引")
        
        # 检索新文档
        results = vector_manager.get_index("demo_docs").as_retriever(similarity_top_k=2).retrieve("AutoRetriever")
        print(f"✅ 检索到 {len(results)} 个相关文档")
        
        return True
        
    except Exception as e:
        print(f"❌ 增强版向量存储演示失败: {e}")
        return False


def demonstrate_enhanced_graph_store():
    """演示增强版图存储管理器"""
    print("\n" + "="*60)
    print("🕸️ 演示增强版图存储管理器")
    print("="*60)
    
    try:
        # 初始化 NetworkX 后端
        graph_manager = EnhancedGraphStoreManager(
            backend_type="networkx",
            db_path="./graph/enhanced_graph.gexf"
        )
        
        print("✅ NetworkX 图存储管理器初始化成功")
        
        # 获取图统计信息
        stats = graph_manager.get_graph_stats("default")
        print(f"📊 图统计信息: {stats}")
        
        # 演示图可视化（如果可能）
        try:
            viz_file = graph_manager.visualize_graph(
                backend_name="default",
                output_file="enhanced_graph_visualization.html"
            )
            if viz_file:
                print(f"✅ 图可视化文件已生成: {viz_file}")
        except Exception as e:
            print(f"⚠️  图可视化失败: {e}")
        
        # 演示多后端支持
        print("\n🔄 演示多后端支持...")
        backends = graph_manager.list_backends()
        print(f"可用后端: {backends}")
        
        return True
        
    except Exception as e:
        print(f"❌ 增强版图存储演示失败: {e}")
        return False


def demonstrate_enhanced_index_service():
    """演示增强版索引服务"""
    print("\n" + "="*60)
    print("⚡ 演示增强版索引服务")
    print("="*60)
    
    try:
        # 初始化增强版索引服务
        enhanced_service = EnhancedIndexService(
            enable_auto_retriever=True,
            enable_query_analysis=True,
            enable_metrics=True
        )
        
        print("✅ 增强版索引服务初始化成功")
        
        # 创建示例文档
        documents = create_sample_documents()
        
        # 分析查询
        test_queries = [
            "Python 编程语言的特点是什么？",
            "机器学习和深度学习的区别",
            "向量数据库和图数据库的比较",
            "RAG 系统如何工作？"
        ]
        
        print("\n🔍 查询分析演示...")
        for query in test_queries:
            analysis = enhanced_service.analyze_query(query)
            print(f"❓ 查询: {query[:30]}...")
            print(f"   意图: {analysis.intent}")
            print(f"   复杂度: {analysis.complexity}")
            print(f"   建议策略: {analysis.suggested_strategies}")
        
        # 检索演示
        print("\n🎯 检索功能演示...")
        
        for query in test_queries[:2]:
            print(f"\n❓ 执行检索: {query[:30]}...")
            
            # 测试不同策略
            strategies = ["auto", "vector", "keyword"]
            
            for strategy in strategies:
                start_time = time.time()
                try:
                    results = enhanced_service.retrieve(
                        query, 
                        collection_name="demo_docs",
                        strategy=strategy,
                        top_k=3
                    )
                    retrieval_time = time.time() - start_time
                    print(f"   策略 {strategy}: {len(results)} 结果, 耗时: {retrieval_time:.3f}s")
                except Exception as e:
                    print(f"   策略 {strategy}: 失败 - {e}")
        
        # 性能指标
        print("\n📈 性能指标...")
        metrics = enhanced_service.get_retrieval_metrics()
        print(f"📊 检索指标: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ 增强版索引服务演示失败: {e}")
        return False


def demonstrate_integration():
    """演示组件集成"""
    print("\n" + "="*60)
    print("🔗 演示组件集成")
    print("="*60)
    
    try:
        # 初始化所有组件
        vector_manager = EnhancedVectorStoreManager(persist_dir="./vector/integration_chroma")
        graph_manager = EnhancedGraphStoreManager(backend_type="networkx")
        enhanced_service = EnhancedIndexService(
            vector_store_manager=vector_manager,
            graph_store_manager=graph_manager
        )
        
        print("✅ 所有组件初始化成功")
        
        # 创建集成测试文档
        documents = create_sample_documents()
        
        # 使用集成服务创建索引
        print("🚀 创建集成索引...")
        index = vector_manager.create_index(documents, "integration_test")
        print("✅ 集成索引创建成功")
        
        # 执行集成检索
        test_query = "Python 和机器学习有什么关系？"
        print(f"🔍 执行集成检索: {test_query}")
        
        results = enhanced_service.retrieve(
            test_query,
            collection_name="integration_test",
            strategy="auto",
            top_k=5
        )
        
        print(f"✅ 检索到 {len(results)} 个结果")
        
        # 显示结果
        for i, result in enumerate(results[:3]):
            print(f"   结果 {i+1}: {result.node.text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 组件集成演示失败: {e}")
        return False


def run_performance_benchmark():
    """运行性能基准测试"""
    print("\n" + "="*60)
    print("🏃 性能基准测试")
    print("="*60)
    
    try:
        # 初始化组件
        vector_manager = EnhancedVectorStoreManager(persist_dir="./vector/benchmark_chroma")
        enhanced_service = EnhancedIndexService()
        
        # 创建大量测试文档
        large_document_count = 100
        print(f"📝 创建 {large_document_count} 个测试文档...")
        
        documents = []
        for i in range(large_document_count):
            doc = LlamaDocument(
                text=f"这是测试文档 {i} 的内容。文档包含一些关键词如 Python、机器学习、人工智能等。"
                     f"文档编号 {i} 帮助测试系统的性能和可扩展性。",
                metadata={"doc_id": i, "category": f"category_{i % 5}"}
            )
            documents.append(doc)
        
        # 测试索引创建性能
        print("🚀 测试索引创建性能...")
        start_time = time.time()
        index = vector_manager.create_index(documents, "benchmark_docs")
        index_time = time.time() - start_time
        print(f"✅ 索引创建耗时: {index_time:.3f}s")
        
        # 测试检索性能
        print("🎯 测试检索性能...")
        test_queries = ["Python", "机器学习", "人工智能", "文档", "测试"]
        
        retrieval_times = []
        for query in test_queries:
            start_time = time.time()
            results = enhanced_service.retrieve(
                query,
                collection_name="benchmark_docs",
                strategy="auto",
                top_k=10
            )
            retrieval_time = time.time() - start_time
            retrieval_times.append(retrieval_time)
            print(f"   查询 '{query}': {len(results)} 结果, 耗时: {retrieval_time:.3f}s")
        
        avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
        print(f"📊 平均检索耗时: {avg_retrieval_time:.3f}s")
        
        # 测试缓存效果
        print("💾 测试缓存效果...")
        cache_test_query = "Python"
        
        # 第一次查询（无缓存）
        start_time = time.time()
        results1 = enhanced_service.retrieve(
            cache_test_query,
            collection_name="benchmark_docs",
            strategy="auto",
            top_k=5
        )
        first_query_time = time.time() - start_time
        
        # 第二次查询（有缓存）
        start_time = time.time()
        results2 = enhanced_service.retrieve(
            cache_test_query,
            collection_name="benchmark_docs",
            strategy="auto",
            top_k=5
        )
        second_query_time = time.time() - start_time
        
        print(f"   首次查询耗时: {first_query_time:.3f}s")
        print(f"   缓存查询耗时: {second_query_time:.3f}s")
        print(f"   缓存加速比: {first_query_time/second_query_time:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能基准测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 开始增强版 RAG 和存储适配演示")
    print("="*60)
    
    # 确保目录存在
    os.makedirs("./vector", exist_ok=True)
    os.makedirs("./graph", exist_ok=True)
    
    # 运行演示
    results = []
    
    results.append(demonstrate_enhanced_vector_store())
    results.append(demonstrate_enhanced_graph_store())
    results.append(demonstrate_enhanced_index_service())
    results.append(demonstrate_integration())
    results.append(run_performance_benchmark())
    
    # 总结
    print("\n" + "="*60)
    print("📋 演示总结")
    print("="*60)
    
    demo_names = [
        "增强版向量存储管理器",
        "增强版图存储管理器", 
        "增强版索引服务",
        "组件集成",
        "性能基准测试"
    ]
    
    for i, (name, success) in enumerate(zip(demo_names, results)):
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{i+1}. {name}: {status}")
    
    total_success = sum(results)
    total_demos = len(results)
    
    print(f"\n📊 总体成功率: {total_success}/{total_demos} ({total_success/total_demos*100:.1f}%)")
    
    if total_success == total_demos:
        print("🎉 所有演示都成功完成！")
    else:
        print("⚠️  部分演示失败，请检查错误信息。")
    
    print("\n💡 提示:")
    print("- 增强版组件基于 LlamaIndex 最佳实践构建")
    print("- 提供了更好的性能、可扩展性和易用性")
    print("- 支持向后兼容，可以逐步迁移现有代码")
    print("- 包含完整的错误处理和性能监控")


if __name__ == "__main__":
    main()