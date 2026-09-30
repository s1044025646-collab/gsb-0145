/**
 * Repeatable end-to-end demonstration using a throwaway database inside tmp/.
 * It generates synthetic data, imports, trains, evaluates and recommends,
 * printing a compact audit trail. Time base and seed are fixed.
 */
import { rmSync } from 'node:fs';
import { openDatabase, closeDatabase } from '../src/db/database.ts';
import { importRatings } from '../src/db/repo.ts';
import { generateRatings } from '../src/model/generator.ts';
import { trainFromCurrent, evaluateAndStore } from '../src/model/service.ts';
import { loadModel } from '../src/model/serialize.ts';
import { explain } from '../src/model/mf.ts';
import { topK } from '../src/model/recommend.ts';

const dbPath = 'tmp/demo-e2e.db';
for (const suffix of ['', '-wal', '-shm']) rmSync(dbPath + suffix, { force: true });

const { db } = openDatabase(dbPath);
const events = generateRatings({
  users: 12,
  items: 10,
  density: 0.5,
  seed: 7,
  startTime: Date.UTC(2024, 0, 1),
  timeSpanMs: 365 * 24 * 3600 * 1000,
  includeSparseUser: true,
});
const imported = importRatings(db, events);
console.log('imported:', imported);

const { modelId, epochLosses } = trainFromCurrent(db, {
  valFraction: 0.2,
  factors: 3,
  epochs: 25,
  learningRate: 0.02,
  regBias: 0.02,
  regFactor: 0.05,
  seed: 42,
  minTrainPerUser: 1,
});
console.log(`model ${modelId}: first SSE ${epochLosses[0]!.toFixed(4)} -> last ${epochLosses.at(-1)!.toFixed(4)}`);

const model = loadModel(db, modelId);
const known = explain(model.data, model.fallback, 'u0', 'i1');
console.log('known pair contribution:', JSON.stringify(known));
const cold = explain(model.data, model.fallback, 'ghost-user', 'ghost-item');
console.log('cold pair fallback:', JSON.stringify(cold));

const report = evaluateAndStore(db, modelId, 5);
console.log('report:', JSON.stringify({ rmse: report.rmse, ndcgAtK: report.ndcgAtK, notes: report.notes }, null, 2));

console.log('top-3 for u0 (all_current excluded):');
console.log(topK(model, 'u0', { k: 3, exclude: 'all_current', currentlySeen: new Set(events.filter((e) => e.userId === 'u0').map((e) => e.itemId)) }));
closeDatabase(db);
console.log(`done; database left at ${dbPath}`);
