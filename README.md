# Local User-Item Collaborative Filtering Backend

A Windows-local, API-only TypeScript/Node.js/SQLite backend. It manages explicit anonymous ratings, timestamped revisions, chronological splits, biased matrix-factorization versions, predictions, Top-K recommendations, score explanations, evaluation reports and model comparisons. There is no web UI, basket-rule mining, or pretrained recommendation service.

## Requirements

- Windows PowerShell
- Node.js 25 or newer (uses built-in `node:sqlite`)
- npm

Install dependencies:

```powershell
npm install
```

## Data And Revision Rules

- Users and items are anonymous string IDs created when ratings are imported.
- A repeated `(userId,itemId)` event is a public revision when its event time is later. Ties use event insertion ID.
- Splits are chronological and immutable. The split cutoff is the event time at the start of the validation tail.
- Training uses only the latest pre-cutoff revision for each pair. Validation uses the latest post-cutoff revision for a pair among events present when the immutable split is created.
- Ratings inserted after split creation never alter that frozen split, even if they later revise a validation pair.
- No validation rating, user/item popularity, global mean, bias, latent vector, or candidate statistic is used to initialize training statistics.
- Missing ratings are never treated as zero scores; the SGD loss sums only observed training ratings.
- Repeated POST with the same `idempotencyKey` returns the original event and does not replace it.

## Matrix Factorization

Prediction before clamping:

```text
p(u,i) = globalMean + userBias[u] + itemBias[i] + dot(userVector[u], itemVector[i])
```

- Training minimizes observed-rating squared error with L2 penalties on biases and factors.
- Initialization uses a deterministic Mulberry32-seeded Gaussian generator; all factors and biases start at zero except small item latent vectors.
- Each epoch processes ratings in fixed `(userId,itemId,eventId)` order.
- Reported per-epoch values are observed SSE, observed-only RMSE, and `SSE + lambda * squared-parameter-norm`.
- Predictions are clamped to the configured rating scale after computing the additive decomposition.
- Cold-start prediction uses only training global mean plus any available training-only bias/factor terms.
- Explanations are additive decompositions and state that low-rank association is not causal evidence.

Top-K candidates are all known items. By default training-seen items are excluded for the requesting user. Ranking uses clamped score, then unclamped score, then lexicographic item ID, so ties are deterministic.

Evaluation reports:

- RMSE over validation ratings.
- NDCG@K on a fixed candidate set: all training items plus that validation user's validation items, excluding that user's training-seen items.
- Unrated fixed-candidate rows use an explicit gain of zero; they are not included in SGD or rating RMSE.
- `emptyRelevantUsers` identifies users whose ideal DCG is zero. A supplied user with no candidates is reported separately as an empty candidate case.

## Test And Demo

```powershell
npm test
npm run demo
```

Tests cover a decomposable/small rating matrix, hand-checked loss and SGD updates, deterministic reproduction, L2 regularization, sparse users, all-tie ordering, cold users/items, repeated ratings, future-information protection, failure paths, transaction rollback, HTTP validation/pagination, archived old versions, and persistence after database reopen.

All SQLite, export, and temporary data files stay under `data/`.

## Start And Stop The API

Start on an automatically assigned loopback port:

```powershell
npm start
```

The process binds only `127.0.0.1` and writes its actual port and project PID to `data/server.json`. Stop only that recorded project process with:

```powershell
npm run stop
```

Optional environment variables:

- `CF_DB_PATH`: SQLite path, default `data/app.sqlite`
- `CF_PORT`: fixed loopback port; omit or set `0` for an automatic port

Database migrations are idempotent and run when a database is opened.

## HTTP API

All POST requests use JSON. Errors have the form:

```json
{"error":{"code":"INVALID_PARAMETER","message":"rating must be between 1 and 5","details":{"field":"rating"}}}
```

Common codes: `INVALID_PARAMETER`, `INVALID_JSON`, `INVALID_SPLIT`, `DUPLICATE_REQUEST`, `PAYLOAD_TOO_LARGE`, `SPLIT_NOT_FOUND`, `MODEL_NOT_FOUND`, `NOT_FOUND`, `INTERNAL_ERROR`.

### Health

```http
GET /health
```

### Import Or Revise A Rating

```powershell
Invoke-RestMethod http://127.0.0.1:<port>/ratings `
  -Method Post -ContentType 'application/json' `
  -Body '{"userId":"u01","itemId":"i01","rating":4.5,"eventTime":"2024-01-01T08:00:00Z","idempotencyKey":"rating-1"}'
```

### Paginate Events

```http
GET /ratings?limit=50&offset=0&includeInactive=false
GET /splits?limit=50&offset=0
GET /models?limit=50&offset=0
GET /reports?limit=50&offset=0
```

### Create A Chronological Split

```json
POST /splits
{"validationCount":4}
```

### Train A Versioned Model

```json
POST /models/train
{"splitId":1,"factors":4,"epochs":20,"learningRate":0.02,"regularization":0.03,"seed":1729}
```

The response includes before/after model counts and every epoch loss.

### Predict And Explain

```json
POST /predict
{"modelId":2,"userId":"u01","itemId":"i09"}
```

### Recommend Unseen Top-K

```json
POST /recommend
{"modelId":2,"userId":"u01","topK":10,"excludeSeen":true}
```

`seenExcluded` in the response states the applied policy.

### Evaluate And Compare

```json
POST /models/evaluate
{"modelId":2,"k":5}

POST /models/compare
{"modelIds":[1,2],"k":5}
```

Reports are stored by model ID. Archived model IDs remain unavailable for prediction, preserving old active versions unchanged.

## CLI

CLI commands print before/after counts for ratings, active ratings, splits, and models. Use a separate database for repeatable scratch runs.

```powershell
npm run cli -- import-json --file data/ratings.json
npm run cli -- import-generated --db data/demo-cli.sqlite --seed 42 --users 8 --items 10
npm run cli -- split --db data/demo-cli.sqlite --validation-count 4
npm run cli -- train --db data/demo-cli.sqlite --split-id 1 --factors 4 --epochs 20 --seed 1729 --regularization 0.03
npm run cli -- predict --db data/demo-cli.sqlite --model-id 1 --user u01 --item i01
npm run cli -- recommend --db data/demo-cli.sqlite --model-id 1 --user u01 --top-k 10
npm run cli -- evaluate --db data/demo-cli.sqlite --model-id 1 --k 5
npm run cli -- compare --db data/demo-cli.sqlite --model-ids 1,2 --k 5
npm run cli -- export --db data/demo-cli.sqlite --model-id 1 --user u01 --top-k 10
```

`import-json` accepts either an array of rating objects or `{"ratings":[...]}`. Exports are written only under `data/exports/`.

## Project Layout

- `src/recommender.ts`: seeded SGD, bias/factor model, explanations, Top-K, RMSE, NDCG
- `src/repository.ts`: validation, idempotency, revisions, split persistence
- `src/modelStore.ts`: immutable model versions, loading, reports, comparison
- `src/server.ts`: local HTTP API
- `src/cli.ts`: command-line workflow
- `src/generator.ts`: deterministic fictional ratings
- `test/`: automated unit, integration and HTTP tests
- `scripts/demo.ts`: repeatable end-to-end demonstration
