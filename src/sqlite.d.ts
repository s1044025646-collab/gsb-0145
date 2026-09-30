/**
 * Minimal ambient declarations for the experimental built-in node:sqlite.
 * Kept local so compilation does not depend on @types/node shipping it.
 */
declare module 'node:sqlite' {
  export interface StatementResultingChanges {
    changes: number | bigint;
    lastInsertRowid: number | bigint;
  }

  export class StatementSync {
    run(...params: unknown[]): StatementResultingChanges;
    get(...params: unknown[]): unknown;
    all(...params: unknown[]): unknown[];
  }

  export interface DatabaseSyncOptions {
    open?: boolean;
    enableForeignKeyConstraints?: boolean;
    enableDoubleQuotedStringLiterals?: boolean;
    allowExtension?: boolean;
  }

  export class DatabaseSync {
    constructor(location: string, options?: DatabaseSyncOptions);
    exec(sql: string): void;
    prepare(sql: string): StatementSync;
    close(): void;
  }
}
