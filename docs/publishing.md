# 发布流程 (Publishing)

本仓库的公开历史与本地私有历史分离。公开分支 `main` 只包含一次干净快照；私有开发历史保存在
本地 `private-history` 分支与离线 bundle 中。

## 绝对不要做

```bash
git push --all        # 会把 private-history 推上去
git push --mirror     # 同上
git push origin private-history
```

只推送 `main`：

```bash
git remote add origin <your-repository-url>
git push -u origin main
```

## 发布前检查

```bash
# 1. 工作树干净
git status --short

# 2. 公开树没有敏感文件
git ls-tree -r --name-only main | grep -E '(^|/)(\.env|conn_wifi\.sh|authorized_keys)$|^archive/|^\.claude/|^tmp/'

# 3. 公开树没有凭据、私有端点或绝对 home 路径
git grep -nE '(sk-[A-Za-z0-9]{16,}|114\.212|ese102|api\.pezayo|/home/[A-Za-z0-9_-]+)' main

# 4. 没有大文件（GitHub 单文件上限 100 MiB）
git ls-tree -r -l main | sort -k4 -nr | head

# 5. 测试与文档
cd robot-agent && pytest -q
cd .. && python scripts/check_markdown_links.py
```

## 创建新的干净快照

如果公开树需要重新生成：

```bash
git branch -m main private-history-$(date +%Y%m%d)
git checkout --orphan main
git add -A
git commit -m "chore: publish clean open-source snapshot"
```

`git add -A` 会遵循 `.gitignore`；提交前必须重复上面的第 2–4 项检查。

## 私密历史备份

私有历史 bundle 包含旧凭据与内网信息，**不得上传到任何公开位置**：

```bash
git bundle create /secure/offline/xlerobot-private-history.bundle --all
```

## 外部工作区

`lerobot_0.3.3_pzy` 与 `lerobot_ycy` 的定制已抽取为 `integrations/lerobot_jetson/` 和
`integrations/lerobot_pi05/` 补丁，可随本仓库发布。`XLeRobot` 是上游硬件参考，不建议直接
vendoring：

- 它有独立历史、上游 remote 与较大的仿真/3D 资产；
- 直接复制会带入上游历史与重复资产；
- 推荐按需克隆，或只发布与你们硬件改动相关的差异。

在外部工作区发布前，同样执行第 2–4 项扫描，并补齐 `NOTICE` 与上游许可证声明。
