import { existsSync, readFileSync, rmSync } from 'node:fs';

const file = 'data/server.json';
if (!existsSync(file)) {
  console.log('no project server metadata found');
  process.exit(0);
}
const metadata = JSON.parse(readFileSync(file, 'utf8')) as { pid: number; port: number };
try {
  process.kill(metadata.pid);
  console.log(`stopped project server pid=${metadata.pid} port=${metadata.port}`);
} catch (error) {
  if (error instanceof Error && 'code' in error && error.code === 'ESRCH') {
    console.log(`project server pid=${metadata.pid} was not running`);
  } else {
    throw error;
  }
} finally {
  rmSync(file, { force: true });
}
