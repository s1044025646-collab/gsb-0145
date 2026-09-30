# 本地用户-物品协同过滤与推荐解释后端

一个只在本机运行的显式评分协同过滤后端：TypeScript + Node.js（原生类型擦除运行）+
内置 `node:sqlite`。提供 HTTP API、命令行（CLI）、虚构评分生成器与自动化测试。

不包含网页前端、购物篮/关联规则挖掘，也不调用任何预训练推荐服务。

## 能力与口径

- 输入为匿名用户 ID、物品 ID、1–5 的显式评分与事件时间（epoch 毫秒）。
- 全量评分修订历史保存在 `rating_events`；去重后的当前评分在 `current_ratings`。
- **修订规则（公开、确定）**：按 `event_time` 最新者胜（last-write-wins）；
  更旧的事件永不覆盖；相同时间保留最早写入值，保证重复导入稳定。
- **数据划分**：按每个用户 `(event_time, event_seq)` 排序，尾部比例进入验证集；
  训练集严格早于验证集，稀疏用户（评分条数 ≤ `minTrainPerUser`）全部留训练集。
- **模型**：带全局均值、用户偏置、物品偏置与潜在因子点积的矩阵分解。
  - 预测：`mu + bu + bi + pu·qi`
  - 仅对**已评分**观测累计平方误差；未评分项不当作 0 分，也不产生梯度。
  - 偏置与因子分别有 L2 正则 `regBias`、`regFactor`。
  - SGD 使用整数种子（mulberry32）与固定更新顺序
    （按 `(user,item,event_time,event_seq)`；单条样本先偏置后因子、因子维度升序）。
  - 偏置初值为 0，因子用种子小随机值；`mu` 等一切统计量**只来自训练集**。
- 输出逐轮训练 SSE、预测分数，以及 `globalMean / userBias / itemBias / dot` 各部分贡献。
- **冷启动回退只用训练数据**并标注来源：未知用户+未知物品用训练全局均值
  （`source: "fallback"`）；单侧冷启动使用训练集物品均值/用户残差（`"mixed"`）。
- 解释明确声明：低秩相似性是**相关性，不是因果关系**。
- **Top-K**：分数降序，同分时物品 ID 升序，排序完全确定；已见物品排除口径
  `exclude=train`（默认，仅排除训练集已见）或 `all_current`（排除当前所有已评分）。
- **评测**：验证集 RMSE；固定候选集（验证物品 ∪ 全部训练物品）下的 NDCG@K，
  相关定义为评分 > 3。空相关集合的用户从均值中剔除并计数说明；无候选物品单独计数；
  验证集为空时 RMSE/NDCG 为 `null`。
- 模型版本不可变：训练新模型不会改变旧模型参数与预测。
- 迁移可重复执行；导入按事务提交，失败整体回滚；提交以幂等键去重。

## 依赖

- Node.js ≥ 22.5（开发环境为 Node 25，使用内置 `node:sqlite` 与原生 TS 运行）。
- 仅开发依赖：`typescript` 与 `@types/node`（`npm install` 安装）。
- 所有数据库与临时文件都位于项目内的 `data/` 与 `tmp/`。

## 安装

```powershell
npm install
```

## 测试

```powershell
npm test                 # 运行 node:test 自动化测试（tests/**/*.test.ts）
npm run typecheck        # TypeScript 严格类型检查
```

测试覆盖：可分解评分矩阵、稀疏用户、全同分、冷启动、重复评分修订、
未来信息泄漏防护、小矩阵损失与梯度手算核对、复现性、正则收缩、幂等导入、
事务回滚、重启后持久化、旧模型预测不变、API 错误码与分页、NDCG 空集处理。
每个测试使用 `tmp/` 下唯一的临时数据库。

## 可重复端到端演示

```powershell
npm run demo
```

该脚本使用固定种子与固定起始时间，在 `tmp/demo-e2e.db` 上完成
生成 → 导入 → 训练 → 解释 → 评测 → 推荐的完整流程。

## 启动 / 停止（仅管理本项目进程）

```powershell
powershell -File scripts/serve.ps1     # 绑定 127.0.0.1，端口由 OS 自动分配
powershell -File scripts/stop.ps1      # 停止 tmp/server.json 中记录的本项目进程
```

启动后实际端口与 PID 写入 `tmp/server.json`。也可以直接运行：

```powershell
npm start
```

监听地址始终为 `http://127.0.0.1:<自动端口>`，不对外暴露。可用环境变量
`CF_DB_PATH` 指定数据库文件（默认 `data/app.db`）。

## CLI

```powershell
# 生成虚构评分（固定种子，可重复）
node --experimental-strip-types --no-warnings src/cli/cli.ts generate --users=20 --items=15 --density=0.4 --seed=7 --out=tmp/gen.json

# 导入（显示操作前后状态；支持幂等键）
node --experimental-strip-types --no-warnings src/cli/cli.ts import --file=tmp/gen.json

# 训练（内部先创建按时间的划分）
node --experimental-strip-types --no-warnings src/cli/cli.ts train --factors=4 --epochs=20 --learningRate=0.01 --regBias=0.02 --regFactor=0.05 --seed=42 --valFraction=0.2

node --experimental-strip-types --no-warnings src/cli/cli.ts losses    --model=1
node --experimental-strip-types --no-warnings src/cli/cli.ts predict   --model=1 --user=u0 --item=i3
node --experimental-strip-types --no-warnings src/cli/cli.ts recommend --model=1 --user=u0 --k=5 --exclude=all_current
node --experimental-strip-types --no-warnings src/cli/cli.ts evaluate  --model=1 --k=10
node --experimental-strip-types --no-warnings src/cli/cli.ts compare   --models=1,2 --k=10
node --experimental-strip-types --no-warnings src/cli/cli.ts export    --model=1 --out=tmp/model1.json
node --experimental-strip-types --no-warnings src/cli/cli.ts status
```

## HTTP API

所有请求/响应均为 JSON。错误响应形如
`{"error":{"code":"VALIDATION_ERROR","message":"...","details":...}}`。

错误码：`VALIDATION_ERROR`(400)、`NOT_FOUND`(404)、`CONFLICT`(409)、
`UNPROCESSABLE`(422)、`INTERNAL`(500)。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康状态与计数 |
| GET | `/ratings?limit=&offset=` | 当前评分分页（`total/limit/offset`） |
| GET | `/events?limit=&offset=` | 修订历史分页 |
| GET | `/items` `/users` | 去重列表分页 |
| POST | `/ratings/import` | 批量导入 `{events:[...]}`，事务+幂等键 |
| POST | `/models/train` | 用当前评分划分并训练，返回模型 ID 与逐轮损失 |
| GET | `/models` `/models/:id` `/models/:id/losses` | 模型与损失 |
| POST | `/models/:id/predict` | `{userId,itemId}` 分数与各部分贡献 |
| POST | `/models/:id/recommend` | `{userId,k,candidates?,exclude?}` Top-K |
| POST | `/models/:id/evaluate` | `{k}` RMSE 与固定候选集 NDCG@K |
| POST | `/models/compare` | `{modelIds:[...],k}` 多模型对比 |
| GET | `/reports` | 已持久化的评测报告 |

导入事件示例：

```json
{
  "events": [
    { "userId": "u0", "itemId": "i3", "rating": 4.5, "eventTime": 1704067200000,
      "idempotencyKey": "uuid-or-natural-key" }
  ]
}
```

幂等：相同 `idempotencyKey` + 相同负载重复提交视为空操作；相同键不同负载返回
`CONFLICT` 且整批回滚。

## 数据与清理

- 数据库默认 `data/app.db`（SQLite WAL），临时产物在 `tmp/`。
- 删除这些文件即完全重置；停止服务请使用 `scripts/stop.ps1`（只操作本项目记录的 PID）。
