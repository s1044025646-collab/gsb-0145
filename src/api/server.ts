import { createServer } from 'node:http';
import { realpathSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { closeDatabase, openDatabase } from '../db/database.ts';
import { createRequestListener } from './app.ts';

/**
 * Bind only to loopback. Port 0 lets the OS pick a free port; the chosen port
 * is written under tmp/ so project tooling can discover it.
 */
function main(): void {
  const dbPath = process.env.CF_DB_PATH ?? 'data/app.db';
  const { db, appliedMigrations } = openDatabase(dbPath);
  const server = createServer(createRequestListener(db));

  server.listen(0, '127.0.0.1', () => {
    const address = server.address();
    if (!address || typeof address === 'string') {
      throw new Error('failed to acquire listening address');
    }
    const port = address.port;
    const info = { port, host: '127.0.0.1', pid: process.pid, dbPath: realpathSync(dbPath) };
    mkdirSync('tmp', { recursive: true });
    writeFileSync('tmp/server.json', JSON.stringify(info, null, 2));
    console.log(`CF backend listening on http://127.0.0.1:${port} (pid ${process.pid})`);
    if (appliedMigrations.length > 0) {
      console.log(`applied migrations: ${appliedMigrations.join(', ')}`);
    }
  });

  const shutdown = () => {
    server.close(() => {
      closeDatabase(db);
      process.exit(0);
    });
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main();
