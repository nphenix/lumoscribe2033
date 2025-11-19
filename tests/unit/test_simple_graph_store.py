"""
简单图存储测试
"""

import os
import shutil
import tempfile


def test_enhanced_graph_store_import():
    """测试导入"""
    try:
        from src.framework.storage.enhanced_graph_store import EnhancedGraphStoreManager
        print("✅ EnhancedGraphStoreManager 导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_enhanced_graph_store_init():
    """测试初始化"""
    try:
        from src.framework.storage.enhanced_graph_store import EnhancedGraphStoreManager

        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_graph.gexf")

        # 初始化管理器
        EnhancedGraphStoreManager(db_path=db_path)
        print("✅ EnhancedGraphStoreManager 初始化成功")

        # 清理
        shutil.rmtree(temp_dir)
        return True
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False

def test_networkx_backend():
    """测试 NetworkX 后端"""
    try:
        from src.framework.storage.enhanced_graph_store import NetworkXGraphBackend

        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_graph.gexf")

        # 初始化后端
        backend = NetworkXGraphBackend(db_path=db_path)
        print("✅ NetworkXGraphBackend 初始化成功")

        # 测试基本功能
        assert backend.graph is not None
        assert backend.db_path == db_path

        # 测试统计信息
        stats = backend.get_graph_stats()
        assert isinstance(stats, dict)
        assert "nodes" in stats
        assert "edges" in stats

        print("✅ NetworkXGraphBackend 基本功能测试成功")

        # 清理
        shutil.rmtree(temp_dir)
        return True
    except Exception as e:
        print(f"❌ NetworkX 后端测试失败: {e}")
        return False

def test_storage_init():
    """测试存储模块初始化"""
    try:
        from src.framework.storage import (
            EnhancedGraphStoreManager,
            EnhancedVectorStoreManager,
        )
        print("✅ 存储模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 存储模块导入失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始简单图存储测试...")

    tests = [
        test_enhanced_graph_store_import,
        test_storage_init,
        test_enhanced_graph_store_init,
        test_networkx_backend,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        print(f"\n🔍 运行测试: {test.__name__}")
        if test():
            passed += 1

    print(f"\n📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print("⚠️ 有测试失败")
