"""
配置管理路由

提供配置验证、环境检查、状态监控等API端点。
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger

from ...framework.shared.config import ConfigManager, Settings

router = APIRouter(prefix="/api/v1/config", tags=["配置管理"])


@router.get("/status", response_model=dict[str, Any])
async def get_config_status():
    """
    获取配置状态

    Returns:
        配置状态信息，包括有效性、环境信息、验证错误等
    """
    try:
        config_manager = ConfigManager()
        status = config_manager.get_config_status()

        logger.info("配置状态查询成功")
        return status
    except Exception as e:
        logger.error(f"获取配置状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate", response_model=dict[str, Any])
async def validate_config():
    """
    验证配置

    Returns:
        验证结果，包括错误列表和警告信息
    """
    try:
        config_manager = ConfigManager()

        # 验证环境配置
        env_errors = config_manager.validate_environment()

        # 验证Settings配置
        settings = Settings()
        settings_errors = settings.validate_config()

        # 合并所有错误
        all_errors = env_errors + settings_errors

        result = {
            "valid": len(all_errors) == 0,
            "environment_errors": env_errors,
            "settings_errors": settings_errors,
            "total_errors": len(all_errors),
            "environment_info": settings.get_environment_info()
        }

        if all_errors:
            logger.warning(f"配置验证发现 {len(all_errors)} 个问题")
            for error in all_errors:
                logger.warning(f"  - {error}")
        else:
            logger.info("✅ 配置验证通过")

        return result
    except Exception as e:
        logger.error(f"配置验证失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/setup-dev")
async def setup_development_environment():
    """
    设置开发环境

    包括生成环境变量模板、创建必要目录等
    """
    try:
        config_manager = ConfigManager()
        config_manager.setup_development_environment()

        logger.info("✅ 开发环境设置完成")
        return {
            "success": True,
            "message": "开发环境设置完成",
            "details": {
                "env_template_generated": True,
                "directories_created": True,
                "validation_checked": True
            }
        }
    except Exception as e:
        logger.error(f"设置开发环境失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/environment", response_model=dict[str, Any])
async def get_environment_info():
    """
    获取环境信息

    Returns:
        详细的环境配置信息
    """
    try:
        settings = Settings()
        env_info = settings.get_environment_info()

        # 添加额外的环境信息
        env_info.update({
            "directories_status": {
                "upload_dir": settings.UPLOAD_DIR,
                "persistence_dir": settings.PERSISTENCE_DIR,
                "vector_dir": settings.VECTOR_DIR,
                "graph_dir": settings.GRAPH_DIR,
                "ide_packages_dir": settings.IDE_PACKAGES_DIR
            },
            "llm_config": {
                "openai_base_url": settings.OPENAI_API_BASE,
                "openai_model": settings.OPENAI_MODEL,
                "ollama_host": settings.OLLAMA_HOST,
                "ollama_model": settings.OLLAMA_MODEL,
                "routing_mode": settings.LLM_ROUTING_MODE
            },
            "database_config": {
                "database_url": settings.DATABASE_URL,
                "chroma_host": settings.CHROMA_HOST,
                "chroma_port": settings.CHROMA_PORT,
                "chroma_path": settings.CHROMA_DB_PATH
            },
            "arq_config": {
                "redis_url": settings.ARQ_REDIS_URL,
                "queue_name": settings.ARQ_QUEUE_NAME,
                "worker_count": settings.ARQ_WORKER_COUNT
            }
        })

        logger.info("环境信息查询成功")
        return env_info
    except Exception as e:
        logger.error(f"获取环境信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/template/env")
async def get_env_template():
    """
    获取环境变量模板

    Returns:
        环境变量模板文件内容
    """
    try:
        config_manager = ConfigManager()
        template_content = config_manager.generate_env_template()

        logger.info("环境变量模板生成成功")
        return {
            "template": template_content,
            "filename": ".env.example",
            "instructions": "将此内容复制到 .env 文件中并根据需要修改配置"
        }
    except Exception as e:
        logger.error(f"生成环境变量模板失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
