import { rmSync } from 'node:fs';
import { openDatabase } from '../src/db.ts';
import { generateRatings } from '../src/generator.ts';
import { createSplit, insertRating } from '../src/repository.ts';
import { compareModels, evaluateAndStoreReport, predictWithModel, recommendWithModel, trainAndStoreModel } from '../src/modelStore.ts';

const dbPath = 'data/demo.sqlite';
rmSync(dbPath, { force: true });
const db = openDatabase(dbPath);
console.log('using', dbPath);
for (const rating of generateRatings(9, 7, 9, 0.75)) {
  insertRating(db, { ...rating, idempotencyKey: `demo-${rating.userId}-${rating.itemId}` });
}
const splitId = createSplit(db, 4).id;
console.log('split', { splitId });
const weak = trainAndStoreModel(db, splitId, { factors: 2, epochs: 5, learningRate: 0.01, regularization: 0.1, seed: 7 });
const strong = trainAndStoreModel(db, splitId, { factors: 4, epochs: 20, learningRate: 0.02, regularization: 0.03, seed: 7 });
console.log('losses', { weak: weak.losses.at(-1), strong: strong.losses.at(-1) });
console.log('prediction', predictWithModel(db, strong.modelId, 'u01', 'i01'));
console.log('top-k', recommendWithModel(db, strong.modelId, 'u01', 5));
console.log('report', evaluateAndStoreReport(db, strong.modelId, 5));
console.log('comparison', compareModels(db, [weak.modelId, strong.modelId], 5));
db.close();
