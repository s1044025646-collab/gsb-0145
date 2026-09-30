import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { openDatabase } from '../db/database.ts';
import {
  countCurrent,
  countEvents,
  distinctItems,
  distinctUsers,
  importRatings,
  itemsForUser,
} from '../db/repo.ts';
import { DEFAULT_CONFIG } from '../model/types.ts';
import type { RatingEvent, TrainConfig } from '../model/types.ts';
import {
  compareModels,
  evaluateAndStore,
  trainFromCurrent,
} from '../model/service.ts';
import { generateRatings } from '../model/generator.ts';
import { loadModel } from '../model/serialize.ts';
import { explain } from '../model/mf.ts';
import { topK } from '../model/recommend.ts';

function status(dbPath: string) {
  const { db } = openDatabase(dbPath);
  const s = {
    events: countEvents(db),
    currentRatings: countCurrent(db),
    users: distinctUsers(db).length,
    items: distinctItems(db).length,
    models: (db.prepare('SELECT COUNT(*) AS n FROM models').get() as { n: number }).n,
  };
  db.close();
  return s;
}

function printStatus(label: string, dbPath: string): void {
  console.log(`${label}:`, JSON.stringify(status(dbPath)));
}

function readJsonFile<T>(path: string): T {
  return JSON.parse(readFileSync(resolve(path), 'utf-8')) as T;
}

function parseConfig(args: string[]): TrainConfig {
  const config = { ...DEFAULT_CONFIG };
  for (const arg of args) {
    const [key, raw] = arg.replace(/^--/, '').split('=');
    if (!key || raw === undefined) throw new Error(`expected --key=value, got ${arg}`);
    if (!(key in config)) throw new Error(`unknown config key: ${key}`);
    (config as Record<string, number>)[key] = Number(raw);
  }
  return config;
}

function usage(): void {
  console.log(`Usage:
  cli status [--db=path]
  cli generate --users=20 --items=15 --density=0.4 --seed=7 --out=tmp/gen.json [--db=path]
  cli import --file=tmp/gen.json [--db=path]
  cli split [--db=path] [--valFraction=0.2 ...]            (shows counts; split is created at train time)
  cli train [--db=path] [--factors=.. --epochs=.. --learningRate=.. --regBias=.. --regFactor=.. --seed=.. --valFraction=..]
  cli losses --model=ID [--db=path]
  cli predict --model=ID --user=u0 --item=i3 [--db=path]
  cli recommend --model=ID --user=u0 --k=5 [--exclude=train|all_current] [--candidates=a,b] [--db=path]
  cli evaluate --model=ID --k=10 [--db=path]
  cli compare --models=1,2 --k=10 [--db=path]
  cli export --model=ID --out=tmp/model.json [--db=path]
`);
}

function flag(args: string[], name: string): string | undefined {
  const prefix = `--${name}=`;
  const hit = args.find((a) => a.startsWith(prefix));
  return hit?.slice(prefix.length);
}

function main(): void {
  const args = process.argv.slice(2);
  const command = args[0];
  const dbPath = flag(args, 'db') ?? process.env.CF_DB_PATH ?? 'data/app.db';
  const rest = args.slice(1).filter((a) => !a.startsWith('--db='));

  if (!command || command === '-h' || command === '--help') {
    usage();
    return;
  }

  const { db } = openDatabase(dbPath);
  try {
    switch (command) {
      case 'status':
        console.log(JSON.stringify(status(dbPath), null, 2));
        break;

      case 'generate': {
        const opts = {
          users: Number(flag(rest, 'users') ?? 20),
          items: Number(flag(rest, 'items') ?? 15),
          density: Number(flag(rest, 'density') ?? 0.4),
          seed: Number(flag(rest, 'seed') ?? 7),
          startTime: Date.UTC(2024, 0, 1),
          timeSpanMs: 365 * 24 * 3600 * 1000,
          includeSparseUser: true,
        };
        const events = generateRatings(opts);
        const out = flag(rest, 'out') ?? 'tmp/gen.json';
        writeFileSync(resolve(out), JSON.stringify(events, null, 2));
        console.log(`generated ${events.length} synthetic ratings -> ${out}`);
        break;
      }

      case 'import': {
        const file = flag(rest, 'file');
        if (!file) throw new Error('--file is required');
        const events = readJsonFile<RatingEvent[]>(file);
        printStatus('before', dbPath);
        console.log('import result:', JSON.stringify(importRatings(db, events)));
        printStatus('after ', dbPath);
        break;
      }

      case 'split':
        console.log('splits are created deterministically during "train"; current status:');
        console.log(JSON.stringify(status(dbPath), null, 2));
        break;

      case 'train': {
        const configArgs = rest.filter((a) => !['--model'].some((p) => a.startsWith(p)));
        const config = parseConfig(configArgs);
        printStatus('before', dbPath);
        const result = trainFromCurrent(db, config);
        console.log('trained:', JSON.stringify(result));
        console.log(`epoch losses (SSE on observed train pairs): ${result.epochLosses.map((n) => n.toFixed(4)).join(', ')}`);
        printStatus('after ', dbPath);
        break;
      }

      case 'losses': {
        const model = loadModel(db, Number(flag(rest, 'model')));
        model.data.stats.epochLosses.forEach((loss, epoch) =>
          console.log(`epoch ${epoch}: ${loss.toFixed(6)}`),
        );
        break;
      }

      case 'predict': {
        const model = loadModel(db, Number(flag(rest, 'model')));
        const userId = flag(rest, 'user');
        const itemId = flag(rest, 'item');
        if (!userId || !itemId) throw new Error('--user and --item are required');
        const result = explain(model.data, model.fallback, userId, itemId);
        console.log(JSON.stringify({ userId, itemId, ...result }, null, 2));
        break;
      }

      case 'recommend': {
        const model = loadModel(db, Number(flag(rest, 'model')));
        const userId = flag(rest, 'user');
        if (!userId) throw new Error('--user is required');
        const candidatesRaw = flag(rest, 'candidates');
        const recs = topK(model, userId, {
          k: Number(flag(rest, 'k') ?? 10),
          candidates: candidatesRaw?.split(','),
          exclude: (flag(rest, 'exclude') as 'train' | 'all_current' | undefined) ?? 'train',
          currentlySeen: itemsForUser(db, userId),
        });
        console.log(`user=${userId} exclude=${flag(rest, 'exclude') ?? 'train'}`);
        console.log('seen:', [...itemsForUser(db, userId)].sort().join(',') || '(none)');
        console.log(JSON.stringify(recs, null, 2));
        break;
      }

      case 'evaluate': {
        const report = evaluateAndStore(
          db,
          Number(flag(rest, 'model')),
          Number(flag(rest, 'k') ?? 10),
        );
        console.log(JSON.stringify(report, null, 2));
        break;
      }

      case 'compare': {
        const ids = (flag(rest, 'models') ?? '').split(',').map(Number).filter(Number.isFinite);
        const reports = compareModels(db, ids, Number(flag(rest, 'k') ?? 10));
        console.log(JSON.stringify(reports, null, 2));
        break;
      }

      case 'export': {
        const model = loadModel(db, Number(flag(rest, 'model')));
        const out = flag(rest, 'out') ?? `tmp/model-${model.id}.json`;
        const payload = {
          id: model.id,
          splitId: model.splitId,
          config: model.config,
          globalMean: model.data.globalMean,
          userBias: Object.fromEntries(model.data.userBias),
          itemBias: Object.fromEntries(model.data.itemBias),
          userFactors: Object.fromEntries(model.data.userFactors),
          itemFactors: Object.fromEntries(model.data.itemFactors),
          epochLosses: model.data.stats.epochLosses,
        };
        writeFileSync(resolve(out), JSON.stringify(payload, null, 2));
        console.log(`exported model ${model.id} -> ${out}`);
        break;
      }

      default:
        usage();
        throw new Error(`unknown command: ${command}`);
    }
  } finally {
    db.close();
  }
}

main();
