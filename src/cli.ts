import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { openDatabase } from './db.js';
import { generateRatings } from './generator.js';
import { createSplit, insertRating } from './repository.js';
import { compareModels, evaluateAndStoreReport, predictWithModel, recommendWithModel, trainAndStoreModel } from './modelStore.js';

function arg(names: string[], fallback?: string): string {
  const args = process.argv.slice(2);
  const command = args[0] ?? '';
  const rest = args.slice(1);
  for (let index = 0; index < rest.length; index += 1) {
    if (names.includes(rest[index]!)) return rest[index + 1] ?? '';
  }
  if (fallback !== undefined) return fallback;
  throw new Error(`${command}: missing option ${names.join(' / ')}`);
}

function count(db: ReturnType<typeof openDatabase>) {
  return {
    ratings: (db.prepare('SELECT count(*) c FROM rating_events').get() as { c: number }).c,
    activeRatings: (db.prepare('SELECT count(*) c FROM rating_events WHERE active=1').get() as { c: number }).c,
    splits: (db.prepare('SELECT count(*) c FROM splits').get() as { c: number }).c,
    models: (db.prepare('SELECT count(*) c FROM models').get() as { c: number }).c,
  };
}

function printState(label: string, db: ReturnType<typeof openDatabase>): void {
  console.log(label, JSON.stringify(count(db)));
}

async function main(): Promise<void> {
  const command = process.argv[2];
  const dbPath = arg(['--db'], 'data/app.sqlite');
  const db = openDatabase(dbPath);
  printState('before:', db);
  try {
    if (command === 'import-json') {
      const file = resolve(arg(['--file']));
      const payload = JSON.parse(readFileSync(file, 'utf8')) as unknown[] | { ratings: unknown[] };
      const ratings = Array.isArray(payload) ? payload : payload.ratings;
      let imported = 0;
      for (const rating of ratings) {
        insertRating(db, rating);
        imported += 1;
      }
      console.log(`imported ${imported} ratings from ${file}`);
    } else if (command === 'import-generated') {
      const seed = Number(arg(['--seed'], '42'));
      const users = Number(arg(['--users'], '8'));
      const items = Number(arg(['--items'], '10'));
      let imported = 0;
      for (const rating of generateRatings(seed, users, items)) {
        insertRating(db, rating);
        imported += 1;
      }
      console.log(`imported ${imported} fictional ratings`);
    } else if (command === 'split') {
      console.log(JSON.stringify(createSplit(db, Number(arg(['--validation-count'], '1')))));
    } else if (command === 'train') {
      const splitId = Number(arg(['--split-id']));
      const epochs = Number(arg(['--epochs'], '20'));
      const factors = Number(arg(['--factors'], '4'));
      const seed = Number(arg(['--seed'], '1729'));
      const regularization = Number(arg(['--regularization'], '0.05'));
      const result = trainAndStoreModel(db, splitId, { epochs, factors, seed, regularization });
      console.log(JSON.stringify({ modelId: result.modelId, firstLoss: result.losses[0], lastLoss: result.losses.at(-1) }));
    } else if (command === 'predict') {
      console.log(JSON.stringify(predictWithModel(db, Number(arg(['--model-id'])), arg(['--user']), arg(['--item'])), null, 2));
    } else if (command === 'recommend') {
      console.log(JSON.stringify(recommendWithModel(db, Number(arg(['--model-id'])), arg(['--user']), Number(arg(['--top-k'], '10'))), null, 2));
    } else if (command === 'evaluate') {
      console.log(JSON.stringify(evaluateAndStoreReport(db, Number(arg(['--model-id'])), Number(arg(['--k'], '5'))), null, 2));
    } else if (command === 'compare') {
      const ids = arg(['--model-ids']).split(',').map((value) => Number(value));
      console.log(JSON.stringify(compareModels(db, ids, Number(arg(['--k'], '5'))), null, 2));
    } else if (command === 'export') {
      const modelId = Number(arg(['--model-id']));
      const out = resolve('data/exports', `model-${modelId}-${new Date().toISOString().replaceAll(':', '-')}.json`);
      mkdirSync('data/exports', { recursive: true });
      const rows = {
        recommendations: recommendWithModel(db, modelId, arg(['--user'], 'u01'), Number(arg(['--top-k'], '10'))),
        report: evaluateAndStoreReport(db, modelId, Number(arg(['--k'], '5'))),
      };
      writeFileSync(out, JSON.stringify(rows, null, 2));
      console.log(out);
    } else {
      console.log('commands: import-generated | split | train | predict | recommend | evaluate | compare | export');
      process.exitCode = 2;
    }
  } finally {
    printState('after: ', db);
    db.close();
  }
}

main().catch((error: Error) => {
  console.error(`ERROR ${error.name}: ${error.message}`);
  process.exitCode = 1;
});
