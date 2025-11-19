"""
tests/ - 测试目录

包含完整的测试套件：
- unit/: 单元测试
- integration/: 集成测试
- contract/: 契约测试
- cli_snapshots/: CLI 快照测试

测试最佳实践：
- pytest 框架
- 异步测试支持
- 固定装置管理
- 覆盖率报告
"""

# 注释掉不存在的子模块，避免导入错误
# from . import cli_snapshots, contract, integration, unit

__all__ = ["unit", "integration", "contract", "cli_snapshots"]
