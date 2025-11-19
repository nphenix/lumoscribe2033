#!/usr/bin/env python3
"""
发布说明生成脚本
用于自动生成 GitHub 发布说明和变更日志
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ReleaseNotesGenerator:
    """发布说明生成器"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.changes_types = {
            "feat": "🚀 新增功能",
            "fix": "🐛 修复内容", 
            "docs": "📚 文档更新",
            "style": "💎 代码格式",
            "refactor": "🔄 代码重构",
            "perf": "⚡ 性能优化",
            "test": "🧪 测试相关",
            "chore": "⚙️ 其他改进",
            "ci": "🔧 CI/CD",
            "build": "📦 构建系统"
        }
    
    def get_git_tags(self) -> List[str]:
        """获取所有 Git 标签"""
        try:
            result = subprocess.run(
                ["git", "tag", "--sort=-version:refname"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            tags = [tag.strip() for tag in result.stdout.strip().split("\n") if tag.strip()]
            return tags
        except subprocess.CalledProcessError as e:
            print(f"❌ 获取 Git 标签失败: {e}")
            return []
    
    def get_latest_tag(self) -> Optional[str]:
        """获取最新标签"""
        tags = self.get_git_tags()
        return tags[0] if tags else None
    
    def get_commits_between_tags(self, start_tag: Optional[str], end_tag: Optional[str] = None) -> List[Dict]:
        """获取两个标签之间的提交"""
        if end_tag is None:
            end_tag = "HEAD"
        
        if start_tag:
            range_spec = f"{start_tag}..{end_tag}"
        else:
            range_spec = end_tag
        
        try:
            # 使用 Conventional Commits 格式解析提交信息
            result = subprocess.run(
                [
                    "git", "log", 
                    "--pretty=format:{\"hash\":\"%H\",\"short_hash\":\"%h\",\"author\":\"%an\",\"email\":\"%ae\",\"date\":\"%ad\",\"message\":\"%s\",\"body\":\"%b\"}",
                    "--date=iso",
                    range_spec
                ],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commits = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    try:
                        commit = json.loads(line)
                        commit["parsed"] = self.parse_conventional_commit(commit["message"])
                        commits.append(commit)
                    except json.JSONDecodeError:
                        continue
            
            return commits
        except subprocess.CalledProcessError as e:
            print(f"❌ 获取提交记录失败: {e}")
            return []
    
    def parse_conventional_commit(self, message: str) -> Dict:
        """解析 Conventional Commits 格式"""
        parts = message.split(":", 1)
        if len(parts) == 2:
            type_part = parts[0].strip()
            description = parts[1].strip()
            
            # 提取类型和范围
            if "(" in type_part and ")" in type_part:
                commit_type = type_part.split("(")[0]
                scope = type_part.split("(")[1].split(")")[0]
            else:
                commit_type = type_part
                scope = None
            
            # 提取破坏性变更标记
            is_breaking = "!" in type_part
            
            return {
                "type": commit_type.lower(),
                "scope": scope,
                "description": description,
                "breaking": is_breaking
            }
        
        return {
            "type": "chore",
            "scope": None,
            "description": message,
            "breaking": False
        }
    
    def categorize_commits(self, commits: List[Dict]) -> Dict[str, List[Dict]]:
        """按类型分类提交"""
        categorized = {category: [] for category in self.changes_types.values()}
        
        for commit in commits:
            parsed = commit.get("parsed", {})
            commit_type = parsed.get("type", "chore")
            category = self.changes_types.get(commit_type, self.changes_types["chore"])
            
            categorized[category].append({
                "hash": commit["short_hash"],
                "author": commit["author"],
                "message": parsed["description"],
                "scope": parsed.get("scope"),
                "breaking": parsed.get("breaking", False)
            })
        
        # 过滤空类别
        return {k: v for k, v in categorized.items() if v}
    
    def generate_changelog_section(self, version: str, commits: List[Dict]) -> str:
        """生成变更日志章节"""
        categorized = self.categorize_commits(commits)
        
        changelog = f"## v{version} ({datetime.now().strftime('%Y-%m-%d')})\n\n"
        
        # 按重要性排序类别
        priority_order = [
            "🚀 新增功能",
            "🐛 修复内容",
            "⚡ 性能优化", 
            "🔄 代码重构",
            "🔧 CI/CD",
            "⚙️ 其他改进",
            "📚 文档更新",
            "🧪 测试相关",
            "💎 代码格式"
        ]
        
        for category in priority_order:
            if category in categorized:
                changelog += f"### {category}\n\n"
                
                for commit in categorized[category]:
                    if commit["breaking"]:
                        changelog += f"- **[破坏性变更]** {commit['message']} (`{commit['hash']}`)\n"
                    elif commit["scope"]:
                        changelog += f"- **[{commit['scope']}]** {commit['message']} (`{commit['hash']}`)\n"
                    else:
                        changelog += f"- {commit['message']} (`{commit['hash']}`)\n"
                
                changelog += "\n"
        
        return changelog
    
    def generate_github_release_body(self, version: str, commits: List[Dict], is_prerelease: bool = False) -> str:
        """生成 GitHub 发布说明"""
        categorized = self.categorize_commits(commits)
        
        release_body = f"""## 📋 版本概述

lumoscribe2033 v{version} - 技术栈预览版发布

**发布日期**: {datetime.now().strftime('%Y-%m-%d')}
**发布类型**: {"预发布" if is_prerelease else "正式发布"}
**主要特性**: 技术栈预览版

"""
        
        # 统计信息
        total_commits = len(commits)
        total_authors = len(set(commit.get("author", "") for commit in commits))
        
        if total_commits > 0:
            release_body += f"### 📊 本次发布统计\n\n"
            release_body += f"- 📝 提交数量: {total_commits}\n"
            release_body += f"- 👥 贡献者: {total_authors}\n"
            release_body += f"- 📅 开发周期: 1 天\n\n"
        
        # 主要变更
        major_changes = []
        if "🚀 新增功能" in categorized:
            major_changes.append(f"新增 {len(categorized['🚀 新增功能'])} 个功能")
        if "🐛 修复内容" in categorized:
            major_changes.append(f"修复 {len(categorized['🐛 修复内容'])} 个问题")
        if "🔧 CI/CD" in categorized:
            major_changes.append(f"优化 {len(categorized['🔧 CI/CD'])} 个流程")
        
        if major_changes:
            release_body += f"### 🎯 主要变更\n\n"
            release_body += f"- {'; '.join(major_changes)}\n\n"
        
        # 功能详情
        if any(cat in categorized for cat in ["🚀 新增功能", "🐛 修复内容"]):
            release_body += f"### 📦 变更详情\n\n"
            
            for category in ["🚀 新增功能", "🐛 修复内容", "🔧 CI/CD", "⚙️ 其他改进"]:
                if category in categorized:
                    release_body += f"#### {category}\n\n"
                    
                    for commit in categorized[category][:10]:  # 限制显示数量
                        if commit["breaking"]:
                            release_body += f"- **[破坏性变更]** {commit['message']}\n"
                        elif commit["scope"]:
                            release_body += f"- **[{commit['scope']}]** {commit['message']}\n"
                        else:
                            release_body += f"- {commit['message']}\n"
                    
                    if len(categorized[category]) > 10:
                        release_body += f"- ... 还有 {len(categorized[category]) - 10} 个提交\n"
                    
                    release_body += "\n"
        
        # 技术栈信息
        release_body += f"""### 🛠️ 技术栈信息

- **Python**: 3.12+
- **FastAPI**: 异步 Web 框架
- **LangChain**: AI 应用开发
- **LlamaIndex**: RAG 解决方案
- **SQLModel**: 数据库 ORM
- **Chroma**: 向量数据库
- **NetworkX**: 图分析
- **平台**: Windows 11

### 🔗 相关链接

- [📖 项目文档](https://github.com/lumoscribe2033/lumoscribe2033#readme)
- [🚀 快速开始](https://github.com/lumoscribe2033/lumoscribe2033/blob/main/specs/001-hybrid-rag-platform/quickstart.md)
- [🤝 贡献指南](https://github.com/lumoscribe2033/lumoscribe2033/blob/main/CONTRIBUTING.md)
- [⚠️ 安全政策](https://github.com/lumoscribe2033/lumoscribe2033/blob/main/SECURITY.md)

---
<p align="center">
  <em>感谢所有贡献者的努力！如果您觉得这个项目有用，请给它一个 ⭐</em>
</p>
"""
        
        return release_body
    
    def save_changelog(self, version: str, content: str, changelog_path: Optional[str] = None):
        """保存变更日志到文件"""
        if changelog_path is None:
            changelog_path = self.repo_path / "CHANGELOG.md"
        else:
            changelog_path = Path(changelog_path)
        
        # 如果文件存在，读取现有内容
        existing_content = ""
        if changelog_path.exists():
            with open(changelog_path, "r", encoding="utf-8") as f:
                existing_content = f.read()
        
        # 新内容添加到顶部
        if existing_content.strip():
            new_content = content + "\n" + existing_content
        else:
            new_content = content
        
        # 写入文件
        with open(changelog_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        print(f"✅ 变更日志已保存到: {changelog_path}")
    
    def generate(self, version: str, previous_tag: Optional[str] = None, 
                 save_changelog: bool = True, is_prerelease: bool = False) -> Dict:
        """生成发布说明"""
        print(f"🚀 开始生成 v{version} 发布说明...")
        
        # 获取提交记录
        commits = self.get_commits_between_tags(previous_tag, None)
        
        if not commits:
            print("❌ 未找到相关提交记录")
            return {"changelog": "", "release_body": ""}
        
        print(f"📋 找到 {len(commits)} 个相关提交")
        
        # 生成变更日志
        changelog = self.generate_changelog_section(version, commits)
        
        # 生成 GitHub 发布说明
        release_body = self.generate_github_release_body(version, commits, is_prerelease)
        
        # 保存变更日志
        if save_changelog:
            self.save_changelog(version, changelog)
        
        return {
            "changelog": changelog,
            "release_body": release_body,
            "commits": commits,
            "stats": {
                "total_commits": len(commits),
                "total_authors": len(set(commit.get("author", "") for commit in commits))
            }
        }


def main():
    parser = argparse.ArgumentParser(description="生成发布说明")
    parser.add_argument("version", help="版本号 (例如: 0.1.0)")
    parser.add_argument("--previous-tag", help="上一个标签")
    parser.add_argument("--no-changelog", action="store_true", help="不保存变更日志文件")
    parser.add_argument("--prerelease", action="store_true", help="标记为预发布")
    parser.add_argument("--changelog-path", help="变更日志文件路径")
    parser.add_argument("--output-format", choices=["markdown", "json"], default="markdown", help="输出格式")
    
    args = parser.parse_args()
    
    generator = ReleaseNotesGenerator()
    
    # 如果没有指定上一个标签，自动获取
    if args.previous_tag is None:
        args.previous_tag = generator.get_latest_tag()
        if args.previous_tag:
            print(f"📝 自动检测到上一个标签: {args.previous_tag}")
    
    # 生成发布说明
    result = generator.generate(
        version=args.version,
        previous_tag=args.previous_tag,
        save_changelog=not args.no_changelog,
        is_prerelease=args.prerelease
    )
    
    # 输出结果
    if args.output_format == "json":
        output = {
            "version": args.version,
            "changelog": result["changelog"],
            "release_body": result["release_body"],
            "stats": result["stats"]
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print("\n" + "="*60)
        print("📋 变更日志")
        print("="*60)
        print(result["changelog"])
        
        print("\n" + "="*60)
        print("🌐 GitHub 发布说明")
        print("="*60)
        print(result["release_body"])
    
    print(f"\n✅ 发布说明生成完成!")
    print(f"📊 统计信息: {result['stats']['total_commits']} 个提交, {result['stats']['total_authors']} 个贡献者")


if __name__ == "__main__":
    main()