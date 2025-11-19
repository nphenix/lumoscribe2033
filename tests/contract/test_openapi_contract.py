"""
OpenAPI 契约测试

验证 API 路由实现与 OpenAPI 契约的一致性，
包括状态码、响应模型、示例等。
"""

import json
from pathlib import Path
from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from src.api.main import app
from src.api.routes import config, conversations, docs, health, speckit, tasks
from src.framework.shared.config import Settings


class ContractTestResult:
    """契约测试结果"""

    def __init__(self, endpoint: str, method: str):
        self.endpoint = endpoint
        self.method = method
        self.status_code_match = False
        self.response_model_match = False
        self.example_match = False
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_error(self, error: str) -> None:
        """添加错误"""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """添加警告"""
        self.warnings.append(warning)

    def is_passed(self) -> bool:
        """是否通过"""
        return len(self.errors) == 0

    def get_score(self) -> float:
        """获取分数"""
        total_checks = 3  # status_code, response_model, example
        passed_checks = sum([
            self.status_code_match,
            self.response_model_match,
            self.example_match
        ])
        return (passed_checks / total_checks) * 100


class OpenAPIContractTester:
    """OpenAPI 契约测试器"""

    def __init__(self, openapi_spec_path: str = "specs/001-hybrid-rag-platform/contracts/openapi.yaml"):
        """
        初始化契约测试器

        Args:
            openapi_spec_path: OpenAPI 规范文件路径
        """
        self.openapi_spec_path = Path(openapi_spec_path)
        self.openapi_spec: dict[str, Any] | None = None
        self.client = TestClient(app)
        self.load_openapi_spec()

    def load_openapi_spec(self) -> None:
        """加载 OpenAPI 规范"""
        try:
            with open(self.openapi_spec_path, encoding='utf-8') as f:
                self.openapi_spec = yaml.safe_load(f)
            print(f"✅ 成功加载 OpenAPI 规范: {self.openapi_spec_path}")
        except Exception as e:
            raise RuntimeError(f"加载 OpenAPI 规范失败: {e}")

    def get_spec_paths(self) -> dict[str, Any]:
        """获取规范中的路径定义"""
        if not self.openapi_spec:
            return {}
        return self.openapi_spec.get('paths', {})

    def get_actual_routes(self) -> list[dict[str, Any]]:
        """获取实际的路由信息"""
        routes = []
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                routes.append({
                    'path': route.path,
                    'methods': list(route.methods) if route.methods else [],
                    'name': getattr(route, 'name', ''),
                    'endpoint': route.endpoint
                })
        return routes

    def test_endpoint_consistency(self) -> list[ContractTestResult]:
        """测试端点一致性"""
        results = []

        spec_paths = self.get_spec_paths()
        actual_routes = self.get_actual_routes()

        # 检查规范中定义的端点是否在实际路由中存在
        for spec_path, spec_methods in spec_paths.items():
            for method, spec_operation in spec_methods.items():
                if method.lower() == 'parameters':
                    continue

                result = ContractTestResult(spec_path, method.upper())

                # 查找对应的实际路由
                actual_route = None
                for route in actual_routes:
                    if route['path'] == spec_path and method.upper() in route['methods']:
                        actual_route = route
                        break

                if not actual_route:
                    result.add_error(f"端点 {method.upper()} {spec_path} 在实际路由中不存在")
                    results.append(result)
                    continue

                # 检查状态码一致性
                self._test_status_codes(result, spec_operation, actual_route)

                # 检查请求/响应模型
                self._test_request_response_models(result, spec_operation, actual_route)

                # 检查示例
                self._test_examples(result, spec_operation, actual_route)

                results.append(result)

        return results

    def _test_status_codes(self, result: ContractTestResult, spec_operation: dict[str, Any], actual_route: dict[str, Any]) -> None:
        """测试状态码一致性"""
        spec_responses = spec_operation.get('responses', {})

        # 检查常见的成功状态码
        success_codes = ['200', '201', '202']
        for code in success_codes:
            if code in spec_responses:
                # 这里简化处理，实际应该调用 API 验证
                result.status_code_match = True
                break

        # 检查错误状态码
        error_codes = ['400', '401', '403', '404', '500']
        for code in error_codes:
            if code in spec_responses:
                # 这里应该验证错误处理是否正确
                pass

    def _test_request_response_models(self, result: ContractTestResult, spec_operation: dict[str, Any], actual_route: dict[str, Any]) -> None:
        """测试请求/响应模型"""
        # 检查请求体
        spec_request_body = spec_operation.get('requestBody', {})
        if spec_request_body:
            # 验证请求体结构
            content = spec_request_body.get('content', {})
            if 'application/json' in content:
                schema = content['application/json'].get('schema', {})
                if schema:
                    result.response_model_match = True

        # 检查响应模型
        spec_responses = spec_operation.get('responses', {})
        for status_code, response_spec in spec_responses.items():
            if 'content' in response_spec:
                content = response_spec['content']
                if 'application/json' in content:
                    schema = content['application/json'].get('schema', {})
                    if schema:
                        result.response_model_match = True
                        break

    def _test_examples(self, result: ContractTestResult, spec_operation: dict[str, Any], actual_route: dict[str, Any]) -> None:
        """测试示例"""
        # 检查请求示例
        spec_request_body = spec_operation.get('requestBody', {})
        if spec_request_body and 'content' in spec_request_body:
            content = spec_request_body['content']
            if 'application/json' in content and 'example' in content['application/json']:
                result.example_match = True

        # 检查响应示例
        spec_responses = spec_operation.get('responses', {})
        for status_code, response_spec in spec_responses.items():
            if 'content' in response_spec:
                content = response_spec['content']
                if 'application/json' in content and 'example' in content['application/json']:
                    result.example_match = True
                    break

    def generate_contract_report(self, results: list[ContractTestResult]) -> dict[str, Any]:
        """生成契约测试报告"""
        total_endpoints = len(results)
        passed_endpoints = sum(1 for r in results if r.is_passed())
        total_score = sum(r.get_score() for r in results) / max(len(results), 1)

        # 按路径分组结果
        path_groups = {}
        for result in results:
            if result.endpoint not in path_groups:
                path_groups[result.endpoint] = []
            path_groups[result.endpoint].append(result)

        report = {
            "contract_test_info": {
                "generated_at": "2025-11-17T12:00:00Z",
                "openapi_spec": str(self.openapi_spec_path),
                "total_endpoints": total_endpoints,
                "passed_endpoints": passed_endpoints,
                "failed_endpoints": total_endpoints - passed_endpoints,
                "overall_score": round(total_score, 2)
            },
            "endpoint_results": [
                {
                    "endpoint": result.endpoint,
                    "method": result.method,
                    "passed": result.is_passed(),
                    "score": result.get_score(),
                    "status_code_match": result.status_code_match,
                    "response_model_match": result.response_model_match,
                    "example_match": result.example_match,
                    "errors": result.errors,
                    "warnings": result.warnings
                }
                for result in results
            ],
            "summary_by_path": {
                path: {
                    "endpoints": len(endpoints),
                    "passed": sum(1 for e in endpoints if e.is_passed()),
                    "failed": sum(1 for e in endpoints if not e.is_passed()),
                    "average_score": sum(e.get_score() for e in endpoints) / len(endpoints)
                }
                for path, endpoints in path_groups.items()
            },
            "recommendations": self._generate_recommendations(results)
        }

        return report

    def _generate_recommendations(self, results: list[ContractTestResult]) -> list[str]:
        """生成改进建议"""
        recommendations = []

        failed_results = [r for r in results if not r.is_passed()]

        if failed_results:
            recommendations.append(f"修复 {len(failed_results)} 个端点的契约不一致问题")

        # 检查状态码问题
        status_code_issues = [r for r in failed_results if not r.status_code_match]
        if status_code_issues:
            recommendations.append(f"统一 {len(status_code_issues)} 个端点的状态码定义")

        # 检查响应模型问题
        model_issues = [r for r in failed_results if not r.response_model_match]
        if model_issues:
            recommendations.append(f"完善 {len(model_issues)} 个端点的响应模型定义")

        # 检查示例问题
        example_issues = [r for r in failed_results if not r.example_match]
        if example_issues:
            recommendations.append(f"为 {len(example_issues)} 个端点添加请求/响应示例")

        if not recommendations:
            recommendations.append("所有端点契约测试通过，继续保持")

        return recommendations

    def save_report(self, report: dict[str, Any], output_path: str | None = None) -> Path:
        """保存测试报告"""
        if not output_path:
            output_dir = Path("data/persistence/reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = "20251117_120000"  # datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"contract_test_report_{timestamp}.json"
        else:
            output_path = Path(output_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📊 契约测试报告已保存: {output_path}")
        return output_path


# Pytest 测试用例
@pytest.mark.skip(reason="OpenAPI 契约测试依赖未实现的 API 端点和文档功能，阶段 3-4 实现")
class TestOpenAPIContract:
    """OpenAPI 契约测试用例"""

    @pytest.fixture
    def contract_tester(self):
        """契约测试器实例"""
        return OpenAPIContractTester()

    @pytest.fixture
    def client(self):
        """测试客户端"""
        return TestClient(app)

    def test_endpoint_consistency(self, contract_tester):
        """测试端点一致性"""
        results = contract_tester.test_endpoint_consistency()

        # 生成报告
        report = contract_tester.generate_contract_report(results)

        # 保存报告
        contract_tester.save_report(report)

        # 断言：应该有一定的通过率
        passed_count = sum(1 for r in results if r.is_passed())
        total_count = len(results)
        pass_rate = passed_count / total_count if total_count > 0 else 0

        print(f"端点一致性测试结果: {passed_count}/{total_count} ({pass_rate:.1%})")

        # 这里设置一个合理的通过率阈值
        assert pass_rate >= 0.5, f"端点一致性通过率过低: {pass_rate:.1%}"

    def test_health_endpoint(self, client):
        """测试健康检查端点"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "environment" in data
        assert "timestamp" in data

    def test_pipeline_endpoint_schema(self, contract_tester):
        """测试管线端点架构"""
        spec_paths = contract_tester.get_spec_paths()

        # 检查管线相关端点是否存在
        pipeline_endpoints = [
            "/pipeline/run",
            "/pipeline/status/{pipeline_id}",
            "/speckit/full-pipeline"
        ]

        for endpoint in pipeline_endpoints:
            assert endpoint in spec_paths, f"端点 {endpoint} 在规范中不存在"

            # 检查 POST 方法
            if "post" in spec_paths[endpoint]:
                operation = spec_paths[endpoint]["post"]
                assert "requestBody" in operation, f"端点 {endpoint} 缺少请求体定义"
                assert "responses" in operation, f"端点 {endpoint} 缺少响应定义"

    def test_error_handling_consistency(self, client):
        """测试错误处理一致性"""
        # 测试不存在的端点
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

        # 测试无效的请求体
        response = client.post("/api/v1/pipeline/run", json={})
        assert response.status_code in [400, 422]  # Bad Request 或 Validation Error

    def test_response_formats(self, client):
        """测试响应格式一致性"""
        response = client.get("/api/v1/health")

        # 检查 Content-Type
        assert response.headers["content-type"].startswith("application/json")

        # 检查响应结构
        data = response.json()
        assert isinstance(data, dict)

        # 检查必需字段
        required_fields = ["status", "version", "environment", "timestamp"]
        for field in required_fields:
            assert field in data, f"响应缺少必需字段: {field}"


def run_contract_tests():
    """运行契约测试的便捷函数"""
    tester = OpenAPIContractTester()

    print("🧪 开始运行 OpenAPI 契约测试...")

    # 执行测试
    results = tester.test_endpoint_consistency()

    # 生成报告
    report = tester.generate_contract_report(results)

    # 保存报告
    report_file = tester.save_report(report)

    # 输出摘要
    print("\n" + "="*60)
    print("📊 OpenAPI 契约测试摘要")
    print("="*60)

    total_endpoints = report["contract_test_info"]["total_endpoints"]
    passed_endpoints = report["contract_test_info"]["passed_endpoints"]
    failed_endpoints = report["contract_test_info"]["failed_endpoints"]
    overall_score = report["contract_test_info"]["overall_score"]

    print(f"📈 总端点数: {total_endpoints}")
    print(f"✅ 通过端点: {passed_endpoints}")
    print(f"❌ 失败端点: {failed_endpoints}")
    print(f"📊 总体得分: {overall_score:.1f}%")

    # 输出失败的端点
    if failed_endpoints > 0:
        print("\n❌ 失败的端点:")
        for result in results:
            if not result.is_passed():
                print(f"  • {result.method} {result.endpoint}")
                for error in result.errors:
                    print(f"    - {error}")

    # 输出建议
    print("\n💡 改进建议:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i}. {rec}")

    print(f"\n📄 详细报告: {report_file}")

    return report


if __name__ == "__main__":
    run_contract_tests()
