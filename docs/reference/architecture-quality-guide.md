<!-- generated: python -m src.cli metadata-injector @ 2025-11-17T00:00:00Z -->
<!-- classification: developer -->

# 架构质量指南

> 基于 2025-11-17 重构经验制定的架构质量标准和最佳实践

## 概述

本文档基于项目重构实践，总结了 DRY、SOLID 原则和设计模式的应用经验，为后续开发提供架构质量指导。

## 核心原则

### DRY (Don't Repeat Yourself) 原则

**目标**: 消除代码重复，提高可维护性

**检查点**:
- [ ] 是否存在重复的数据模型定义？→ 统一到 `shared/models.py`
- [ ] 是否存在重复的业务逻辑？→ 提取为基类或工具函数
- [ ] 是否存在重复的错误处理模式？→ 使用上下文管理器或装饰器
- [ ] 代码重复度是否 <5%？

**实践示例**:
```python
# ❌ 违反 DRY：重复的 RetrievalMetrics 定义
# 在 enhanced_index_service.py 和 llamaindex_service.py 中都有定义

# ✅ 遵循 DRY：统一到 shared/models.py
from src.framework.shared.models import RetrievalMetrics
```

### SOLID 原则

#### 单一职责原则 (SRP)

**目标**: 每个类/模块只负责一个明确的功能领域

**检查点**:
- [ ] 类的方法是否都在处理同一个业务领域？
- [ ] 类是否承担了多个职责（如配置管理 + 环境验证 + 模板生成）？
- [ ] 是否需要拆分为多个职责单一的组件？

**实践示例**:
```python
# ❌ 违反 SRP：ConfigManager 承担多个职责
class ConfigManager:
    def validate_environment(self): ...  # 环境验证
    def generate_template(self): ...     # 模板生成
    def get_model_config(self): ...      # 模型配置
    def get_routing_config(self): ...    # 路由配置

# ✅ 遵循 SRP：职责分离
class EnvironmentValidator:
    def validate_environment(self): ...

class ConfigTemplateGenerator:
    def generate_template(self): ...

class ModelConfigManager:
    def get_model_config(self): ...

class ConfigManager:  # 协调者
    def __init__(self):
        self.env_validator = EnvironmentValidator()
        self.template_generator = ConfigTemplateGenerator()
        self.model_manager = ModelConfigManager()
```

#### 开闭原则 (OCP)

**目标**: 对扩展开放，对修改关闭

**检查点**:
- [ ] 是否需要添加新功能？→ 使用策略模式而非修改现有代码
- [ ] 是否需要支持新的实现？→ 使用适配器模式统一接口
- [ ] 是否需要创建新类型？→ 使用工厂模式而非硬编码

**实践示例**:
```python
# ❌ 违反 OCP：硬编码的条件分支
def retrieve(self, strategy: str):
    if strategy == "vector":
        return self._vector_retrieval()
    elif strategy == "graph":
        return self._graph_retrieval()
    # 添加新策略需要修改此方法

# ✅ 遵循 OCP：策略模式
class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(self): ...

class VectorRetrievalStrategy(RetrievalStrategy): ...
class GraphRetrievalStrategy(RetrievalStrategy): ...

# 添加新策略无需修改现有代码
class HybridRetrievalStrategy(RetrievalStrategy): ...
```

#### 依赖倒置原则 (DIP)

**目标**: 依赖抽象而非具体实现

**检查点**:
- [ ] 客户端代码是否依赖具体类？→ 改为依赖抽象接口
- [ ] 是否可以通过接口统一不同实现？→ 使用适配器模式

**实践示例**:
```python
# ❌ 违反 DIP：依赖具体实现
def use_vector_store():
    store = VectorStoreManager()  # 直接依赖具体类
    store.add_documents(...)

# ✅ 遵循 DIP：依赖抽象接口
def use_vector_store(store: IVectorStore):  # 依赖接口
    store.add_documents(...)

# 客户端无需关心具体实现
store = VectorStore(implementation="basic")  # 或 "enhanced"
```

## 设计模式应用指南

### 策略模式 (Strategy Pattern)

**适用场景**: 需要支持多种算法或行为，且可以动态切换

**示例**: 检索策略、元数据注入策略

```python
# 定义策略接口
class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(self, query: str, ...): ...

# 实现具体策略
class VectorRetrievalStrategy(RetrievalStrategy): ...
class HybridRetrievalStrategy(RetrievalStrategy): ...

# 使用工厂创建策略
strategy = RetrievalStrategyFactory.create_strategy("hybrid")
results = strategy.retrieve(query, ...)
```

### 适配器模式 (Adapter Pattern)

**适用场景**: 需要统一不同实现的接口

**示例**: 向量存储适配器、元数据注入适配器

```python
# 定义统一接口
class IVectorStore(ABC):
    @abstractmethod
    def add_documents(self, ...): ...

# 适配器实现
class BasicVectorStoreAdapter(IVectorStore):
    def __init__(self, base_manager):
        self.manager = base_manager
    def add_documents(self, ...):
        return self.manager.add_documents(...)

# 统一接口
class VectorStore:
    def __init__(self, implementation="basic"):
        if implementation == "basic":
            manager = VectorStoreManager()
            self.adapter = BasicVectorStoreAdapter(manager)
```

### 工厂模式 (Factory Pattern)

**适用场景**: 需要根据参数创建不同类型的对象

**示例**: 检索策略工厂、适配器工厂

```python
class RetrievalStrategyFactory:
    _strategy_classes = {
        "vector": VectorRetrievalStrategy,
        "hybrid": HybridRetrievalStrategy,
    }
    
    @classmethod
    def create_strategy(cls, name: str, ...):
        return cls._strategy_classes[name](...)
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class):
        cls._strategy_classes[name] = strategy_class
```

### 组合模式 (Composition Pattern)

**适用场景**: 需要将多个组件组合成更复杂的对象

**示例**: ConfigManager 组合各配置组件

```python
class ConfigManager:
    def __init__(self):
        # 组合多个单一职责的组件
        self.model_manager = ModelConfigManager()
        self.env_validator = EnvironmentValidator()
        self.template_generator = ConfigTemplateGenerator()
    
    # 委托方法
    def get_model_by_name(self, name: str):
        return self.model_manager.get_model_by_name(name)
```

## 重构检查清单

在开发新功能或修改现有功能时，使用此清单确保架构质量：

### 代码重复检查
- [ ] 是否存在重复的数据模型？→ 统一到 `shared/models.py`
- [ ] 是否存在重复的业务逻辑？→ 提取为基类或工具函数
- [ ] 是否存在重复的错误处理？→ 使用 `ErrorContext` 上下文管理器
- [ ] 代码重复度是否 <5%？

### 单一职责检查
- [ ] 类的方法是否都在处理同一个业务领域？
- [ ] 类是否承担了多个职责？
- [ ] 是否需要拆分为多个组件？

### 扩展性检查
- [ ] 是否需要支持新的实现？→ 使用策略模式
- [ ] 是否需要统一不同接口？→ 使用适配器模式
- [ ] 是否需要动态创建对象？→ 使用工厂模式

### 接口设计检查
- [ ] 是否定义了清晰的抽象接口？
- [ ] 客户端是否依赖抽象而非具体实现？
- [ ] 不同实现是否通过统一接口暴露？

## 常见反模式及解决方案

### 反模式 1: 上帝类 (God Class)

**症状**: 一个类承担了太多职责，方法数量 >20

**解决方案**: 使用组合模式拆分为多个单一职责的组件

### 反模式 2: 重复代码 (Duplicated Code)

**症状**: 相同的代码模式在多处出现

**解决方案**: 提取为基类、工具函数或使用装饰器

### 反模式 3: 硬编码条件分支

**症状**: 使用大量 if/else 处理不同情况

**解决方案**: 使用策略模式或工厂模式

### 反模式 4: 紧耦合

**症状**: 类直接依赖具体实现，难以替换

**解决方案**: 定义抽象接口，使用依赖注入

## 代码审查要点

在代码审查时，重点关注：

1. **DRY 违反**: 是否存在重复代码？
2. **SRP 违反**: 类是否承担了多个职责？
3. **OCP 违反**: 添加新功能是否需要修改现有代码？
4. **DIP 违反**: 是否依赖具体实现而非抽象？
5. **设计模式**: 在需要扩展性时是否使用了设计模式？

## 参考资源

- 项目重构经验总结: `docs/internal/architecture-refactoring-2025-11-17.md` (待创建)
- 设计模式示例: `src/framework/rag/retrieval_strategy.py`
- 适配器模式示例: `src/framework/storage/vector_store_interface.py`
- 组合模式示例: `src/framework/shared/config_components.py`

---

*本文档基于 2025-11-17 架构重构实践制定，将随项目演进持续更新。*

