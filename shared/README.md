# Shared Context

Skill 间信息共享目录。

## 文件说明

| 文件 | 说明 |
|------|------|
| `context.template.yaml` | context 模板文件，仅用于初始化容器内的 context.yaml |

## 多任务隔离设计

项目目录下的 `context.template.yaml` 是**模板**，不是运行时状态文件。

每个迁移任务启动时，`setup_workspace.sh` 会将模板复制到容器内 `/flagos-workspace/shared/context.yaml`，作为该任务的独立 context。多个任务并发执行时，各容器内的 context 互不干扰。

```
context.template.yaml (项目目录，只读模板)
        │
        ├── docker cp → 容器A:/flagos-workspace/shared/context.yaml (任务A独立)
        ├── docker cp → 容器B:/flagos-workspace/shared/context.yaml (任务B独立)
        └── docker cp → 容器C:/flagos-workspace/shared/context.yaml (任务C独立)
```

## 使用约定

### 写入方 (上游 Skill)

- 通过 `docker exec` 在容器内写入 `/flagos-workspace/shared/context.yaml`
- 必须更新 `metadata.created_by` 和 `metadata.updated_at`

### 读取方 (下游 Skill)

- 通过 `docker exec` 从容器内读取 `/flagos-workspace/shared/context.yaml`
- 如果容器内 context.yaml 为空或不存在，提示用户手动配置

### 宿主机归档

- 步骤6发布阶段和兜底同步会将容器内 context 回传到宿主机：`/data/flagos-workspace/<model>/config/context_snapshot.yaml`
- 全流程结束时回传最终状态：`/data/flagos-workspace/<model>/config/context_final.yaml`

## 数据流

```
context.template.yaml ──初始化──> 容器内 context.yaml ──读写──> 各 Skill
                                                        ──回传──> 宿主机 context_snapshot.yaml / context_final.yaml
```
