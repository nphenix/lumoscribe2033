#!/usr/bin/env python3
# generated: python -m src.cli metadata-injector @ 2025-11-19T13:05:21Z
# classification: developer
"""
LlamaIndex RAG 系统演示脚本

基于 LlamaIndex 最佳实践演示完整的 RAG 功能。
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from src.framework.rag.llamaindex_service import LlamaIndexService


async def demonstrate_llamaindex_rag():
    """演示 LlamaIndex RAG 系统功能"""
    print("🚀 开始 LlamaIndex RAG 系统演示...")
    
    try:
        # 1. 准备演示数据
        print("\n📝 准备演示数据...")
        
        demo_documents = [
            Document(
                text="""
                Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。
                Python 以其简洁易读的语法而闻名，支持多种编程范式，包括
                面向对象、命令式、函数式和过程式编程。Python 的设计哲学强调
                代码的可读性和简洁的语法结构。
                """,
                metadata={
                    "title": "Python 编程语言介绍",
                    "category": "programming",
                    "language": "python",
                    "tags": ["python", "programming", "language"]
                }
            ),
            Document(
                text="""
                机器学习是人工智能的一个子领域，专注于开发算法和统计模型，
                使计算机系统能够执行任务而无需明确的指令。机器学习算法
                基于样本数据（称为"训练数据"）构建数学模型，用于做出预测
                或决策，而无需为任务编程明确的指令。主要分为监督学习、
                无监督学习和强化学习三种类型。
                """,
                metadata={
                    "title": "机器学习基础概念",
                    "category": "artificial_intelligence",
                    "field": "machine_learning",
                    "tags": ["machine_learning", "ai", "algorithms"]
                }
            ),
            Document(
                text="""
                神经网络是受生物神经网络启发的计算系统，通过估计相互关联的
                单元（称为神经元）之间的复杂关系来处理信息。神经网络也称为
                人工神经网络（ANN）或连接主义系统。深度学习是机器学习的一个
                子集，使用包含多个隐藏层的神经网络。
                """,
                metadata={
                    "title": "神经网络和深度学习",
                    "category": "deep_learning",
                    "field": "neural_networks",
                    "tags": ["neural_networks", "deep_learning", "ai"]
                }
            ),
            Document(
                text="""
                数据库管理系统（DBMS）是一种软件应用程序，用于与数据库用户、 
                其他应用程序和数据库本身交互。DBMS 的主要目标是为数据的
                存储、检索和管理提供一种方式，同时确保数据的安全性、
                完整性和一致性。常见的 DBMS 包括 MySQL、PostgreSQL、MongoDB 等。
                """,
                metadata={
                    "title": "数据库管理系统概述",
                    "category": "database",
                    "type": "management_system",
                    "tags": ["database", "dbms", "storage"]
                }
            ),
            Document(
                text="""
                Web 开发是创建 Web 应用程序的过程，涉及 Web 设计、Web 内容
                开发、客户端/服务器端脚本、Web 应用程序开发和 Web 服务器
                配置。Web 开发的范围从创建简单的静态页面到复杂的 Web
                应用程序、电子政务、电子商务、Web 门户等。主要技术包括
                HTML、CSS、JavaScript、React、Vue 等。
                """,
                metadata={
                    "title": "Web 开发技术",
                    "category": "web_development",
                    "type": "frontend_backend",
                    "tags": ["web_development", "frontend", "backend"]
                }
            ),
            Document(
                text="""
                自然语言处理（NLP）是人工智能的一个分支，专注于计算机与人类语言之间的交互。
                NLP 技术使计算机能够理解、解释、操作和生成人类语言。应用包括机器翻译、
                情感分析、语音识别、聊天机器人等。近年来，基于 Transformer 的模型
                如 BERT、GPT 系列在 NLP 任务中取得了突破性进展。
                """,
                metadata={
                    "title": "自然语言处理技术",
                    "category": "nlp",
                    "field": "computational_linguistics",
                    "tags": ["nlp", "transformer", "bert"]
                }
            )
        ]
        
        print(f"✅ 准备了 {len(demo_documents)} 个演示文档")
        
        # 2. 初始化 LlamaIndex 服务
        print("\n🔧 初始化 LlamaIndex 服务...")
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 初始化服务（使用模拟的嵌入和 LLM）
            service = LlamaIndexService(
                persist_dir=f"{temp_dir}/llamaindex_storage",
                chunk_size=512,
                chunk_overlap=100,
            )
            
            print("✅ LlamaIndex 服务已初始化")
            
            # 3. 创建索引
            print("\n📊 创建索引...")
            
            # 创建主要索引
            index = await service.create_index_from_documents(
                demo_documents,
                index_name="demo_index",
                use_auto_retriever=True,
                similarity_top_k=8,
            )
            
            print(f"✅ 索引创建成功")
            print(f"   📈 文档数量: {len(index.docstore.docs)}")
            print(f"   🔗 节点数量: {len(index.index_struct.nodes)}")
            
            # 4. 演示查询功能
            print("\n🔍 演示查询功能...")
            
            test_queries = [
                "什么是 Python 编程语言？",
                "机器学习有哪些类型？",
                "神经网络和深度学习有什么区别？",
                "数据库管理系统的作用是什么？",
                "Web 开发包括哪些技术？",
                "自然语言处理的应用有哪些？"
            ]
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n   问题 {i}: {query}")
                
                try:
                    # 执行查询
                    response = await service.query(
                        query_str=query,
                        index_name="demo_index",
                        similarity_top_k=5,
                    )
                    
                    print(f"   💡 回答: {str(response)[:100]}...")
                    
                    # 检索相关文档
                    retrieved_docs = await service.retrieve(
                        query_str=query,
                        index_name="demo_index",
                        similarity_top_k=3,
                    )
                    
                    print(f"   📄 检索到 {len(retrieved_docs)} 个相关文档")
                    
                    if retrieved_docs:
                        first_doc_title = retrieved_docs[0].metadata.get("title", "未知标题")
                        print(f"   📋 第一个相关文档: {first_doc_title}")
                        
                except Exception as e:
                    print(f"   ❌ 查询失败: {e}")
            
            # 5. 演示混合索引
            print("\n🔀 演示混合索引功能...")
            
            try:
                hybrid_index = await service.create_hybrid_index(
                    demo_documents[:3],  # 使用前3个文档
                    index_name="hybrid_demo",
                )
                print("✅ 混合索引创建成功")
                
                # 测试混合查询
                hybrid_response = await service.query(
                    query_str="Python 和机器学习的关系",
                    index_name="hybrid_demo",
                )
                print(f"   🧮 混合查询结果: {str(hybrid_response)[:80]}...")
                
            except Exception as e:
                print(f"   ⚠️ 混合索引演示跳过: {e}")
            
            # 6. 演示索引管理功能
            print("\n🗂️ 演示索引管理功能...")
            
            # 列出所有索引
            indices = service.list_indices()
            print(f"📊 可用索引: {indices}")
            
            # 获取索引信息
            index_info = await service.get_index_info("demo_index")
            print(f"📈 主索引信息:")
            print(f"   名称: {index_info.get('name', 'N/A')}")
            print(f"   文档数: {index_info.get('document_count', 'N/A')}")
            print(f"   节点数: {index_info.get('node_count', 'N/A')}")
            
            # 7. 演示文档管理
            print("\n📄 演示文档管理功能...")
            
            # 添加新文档
            new_doc = Document(
                text="区块链是一种分布式数据库技术，通过加密确保数据的安全性和完整性。",
                metadata={
                    "title": "区块链技术介绍",
                    "category": "blockchain",
                    "tags": ["blockchain", "distributed", "security"]
                }
            )
            
            try:
                await service.add_documents(
                    [new_doc],
                    index_name="demo_index",
                )
                print("✅ 新文档已添加到索引")
                
                # 验证添加
                updated_info = await service.get_index_info("demo_index")
                print(f"📈 更新后的文档数量: {updated_info.get('document_count', 'N/A')}")
                
            except Exception as e:
                print(f"   ⚠️ 文档添加功能演示跳过: {e}")
            
            # 8. 演示高级检索功能
            print("\n🎯 演示高级检索功能...")
            
            try:
                # 获取相关节点（用于调试）
                nodes = await service.get_relevant_nodes(
                    "Python 编程",
                    index_name="demo_index",
                    top_k=3,
                )
                print(f"🔍 获取到 {len(nodes)} 个相关节点")
                
                # 检索特定类型的文档
                python_docs = await service.retrieve(
                    query_str="Python",
                    index_name="demo_index",
                    similarity_top_k=5,
                )
                print(f"🐍 找到 {len(python_docs)} 个与 Python 相关的文档")
                
            except Exception as e:
                print(f"   ⚠️ 高级检索功能演示跳过: {e}")
            
            print("\n🎉 LlamaIndex RAG 演示完成！")
            print("\n📚 总结:")
            print("   ✅ 基于 LlamaIndex 最佳实践构建")
            print("   ✅ 支持文档自动分割和索引创建")
            print("   ✅ 提供向量检索和自动检索器")
            print("   ✅ 支持混合索引（向量 + 关键词）")
            print("   ✅ 实现持久化存储和索引管理")
            print("   ✅ 提供高级查询和检索功能")
            print("   ✅ 支持动态文档添加和删除")
            print("   ✅ 包含完整的错误处理和日志记录")
            
            # 9. 对比传统方法的优势
            print("\n🚀 LlamaIndex 相比传统方法的优势:")
            print("   📊 自动文档分割和节点管理")
            print("   🔍 内置的 AutoRetriever 智能检索")
            print("   🎯 多种检索策略和后处理器")
            print("   🔄 无缝的持久化和加载机制")
            print("   🌐 支持多模态和混合检索")
            print("   🛠️ 丰富的配置选项和扩展点")
            print("   📈 内置的性能优化和缓存机制")
            
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            await service.close()
            print("\n🧹 资源已清理")
        except Exception as e:
            print(f"\n⚠️ 清理资源时发生错误: {e}")


async def demonstrate_file_based_indexing():
    """演示基于文件的索引创建"""
    print("\n📁 演示基于文件的索引创建...")
    
    try:
        # 创建临时文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建一些示例文件
            test_files = {
                "python_guide.md": """
                # Python 编程指南
                
                Python 是一种高级编程语言，具有简洁的语法和强大的功能。
                Python 广泛应用于 Web 开发、数据科学、人工智能等领域。
                
                ## 主要特性
                
                - 简洁易读的语法
                - 丰富的标准库
                - 强大的社区支持
                - 跨平台兼容性
                """,
                "machine_learning.txt": """
                机器学习技术概述
                
                机器学习是人工智能的核心技术之一，主要包括：
                
                1. 监督学习：使用标记数据进行训练
                2. 无监督学习：发现数据中的模式
                3. 强化学习：通过试错学习最优策略
                
                应用领域包括图像识别、自然语言处理、推荐系统等。
                """,
                "web_development.html": """
                <html>
                <head><title>Web 开发技术</title></head>
                <body>
                <h1>Web 开发技术指南</h1>
                <p>Web 开发涉及前端和后端技术的结合。</p>
                <ul>
                <li>HTML: 结构标记</li>
                <li>CSS: 样式设计</li>
                <li>JavaScript: 交互逻辑</li>
                </ul>
                </body>
                </html>
                """
            }
            
            # 写入测试文件
            for filename, content in test_files.items():
                file_path = Path(temp_dir) / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            print(f"✅ 创建了 {len(test_files)} 个测试文件")
            
            # 创建基于文件的索引
            service = LlamaIndexService(
                persist_dir=f"{temp_dir}/file_storage",
            )
            
            # 从目录创建索引
            index = await service.create_index_from_directory(
                directory_path=temp_dir,
                index_name="file_based_index",
                file_extensions=[".md", ".txt", ".html"],
                similarity_top_k=6,
            )
            
            print(f"✅ 基于文件的索引创建成功")
            print(f"   📈 索引中的文档数量: {len(index.docstore.docs)}")
            
            # 测试查询
            test_queries = [
                "Python 的主要特性是什么？",
                "机器学习有哪些类型？",
                "Web 开发包括哪些技术？"
            ]
            
            for query in test_queries:
                response = await service.query(
                    query_str=query,
                    index_name="file_based_index",
                )
                print(f"   📋 问题: {query}")
                print(f"   💡 答案: {str(response)[:100]}...")
            
            await service.close()
            
    except Exception as e:
        print(f"❌ 文件索引演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🎯 LlamaIndex RAG 系统演示")
    print("=" * 50)
    
    # 运行演示
    asyncio.run(demonstrate_llamaindex_rag())
    
    # 演示文件索引
    asyncio.run(demonstrate_file_based_indexing())