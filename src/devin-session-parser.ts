/**
 * Devin session parser — reads Devin Desktop/CLI sessions directly from its
 * local SQLite store at <appData>/Devin/cli/sessions.db.
 *
 * Schema (observed, Devin Desktop 2026):
 *   sessions(id, working_directory, backend_type, model, agent_mode,
 *            created_at, last_activity_at, title, main_chain_id, hidden, ...)
 *   message_nodes(row_id, session_id, node_id, parent_node_id, chat_message,
 *                 created_at, ...)
 * chat_message is a JSON blob: { message_id, role, content, tool_calls?, ... }
 * with role in {user, assistant, tool, system}. Timestamps are unix seconds.
 *
 * Consistent with prose's tool-result exclusion policy: `tool` messages and
 * `tool_calls` payloads are dropped — tool output (file dumps, command blobs)
 * pollutes semantic memory without adding intent.
 */

import { existsSync, statSync } from 'fs';
import { join, basename } from 'path';
import { homedir } from 'os';
import type { Conversation, Message, SessionFile } from './session-parser.js';
import { openReadonly, sqliteTs, type ReadonlyDb } from './sqlite.js';
import { sanitizePath } from './memory.js';

// ============================================================================
// Discovery
// ============================================================================

/**
 * Candidate locations for Devin's session database. Devin is an Electron app;
 * its userData dir follows the per-platform appData convention.
 */
export function getDevinDbPaths(): string[] {
  const home = homedir();
  const paths: string[] = [];
  if (process.platform === 'win32') {
    const appData = process.env.APPDATA || join(home, 'AppData', 'Roaming');
    paths.push(join(appData, 'Devin', 'cli', 'sessions.db'));
  } else if (process.platform === 'darwin') {
    paths.push(join(home, 'Library', 'Application Support', 'Devin', 'cli', 'sessions.db'));
  } else {
    const config = process.env.XDG_CONFIG_HOME || join(home, '.config');
    paths.push(join(config, 'Devin', 'cli', 'sessions.db'));
  }
  return paths;
}

export function getDevinDbPath(): string | null {
  for (const p of getDevinDbPaths()) {
    if (existsSync(p)) return p;
  }
  return null;
}

interface DevinSessionRow {
  id: string;
  working_directory: string;
  title: string | null;
  created_at: number;
  last_activity_at: number;
  hidden: number;
}

/**
 * Project naming/filtering mirrors the Codex parser: the sanitized cwd with a
 * leading dash, matched exactly or as a suffix so `-Users-x-repo` hits `repo`.
 */
function getDevinProjectName(cwd?: string): string | null {
  if (!cwd || typeof cwd !== 'string') return null;
  return `-${sanitizePath(cwd)}`;
}

function matchesProjectFilter(projectName: string, projectPath: string): boolean {
  const normalizedProjectPath = sanitizePath(projectPath);
  const normalizedProjectName = projectName.replace(/^-/, '');
  return normalizedProjectName === normalizedProjectPath ||
    normalizedProjectName.endsWith(`-${normalizedProjectPath}`);
}

/**
 * Discover Devin sessions. `projectPath` filters on the session's recorded
 * working_directory — exact for Devin since the DB stores the real cwd.
 */
export function discoverDevinSessionFiles(projectPath?: string): SessionFile[] {
  const dbPath = getDevinDbPath();
  if (!dbPath) return [];

  const db = openReadonly(dbPath);
  if (!db) return [];

  try {
    const rows = db
      .prepare(
        `SELECT id, working_directory, title, created_at, last_activity_at, hidden
         FROM sessions ORDER BY last_activity_at DESC`
      )
      .all() as unknown as DevinSessionRow[];

    const fileSize = statSync(dbPath).size;
    const now = new Date();
    const out: SessionFile[] = [];

    for (const row of rows) {
      if (row.hidden) continue;
      const cwd = row.working_directory || '';
      const projectName = getDevinProjectName(cwd);
      if (projectPath) {
        if (!projectName || !matchesProjectFilter(projectName, projectPath)) continue;
      }

      out.push({
        path: dbPath,
        sessionId: row.id,
        project: projectName || 'devin',
        modifiedTime: sqliteTs(row.last_activity_at, now),
        fileSize,
        sourceType: 'devin',
        cwd: cwd || undefined,
      });
    }
    return out;
  } catch {
    return [];
  } finally {
    try { db.close(); } catch {}
  }
}

// ============================================================================
// Parsing
// ============================================================================

// Cheap prescan over raw chat_message text — avoids JSON.parse on replayed
// chain copies and tool/system rows entirely.
const ROLE_RE = /"role"\s*:\s*"(\w+)"/;
const ID_RE = /"message_id"\s*:\s*"([^"]+)"/;

interface DevinChatMessage {
  message_id?: string;
  role?: string;
  content?: unknown;
  tool_calls?: unknown;
  created_at?: number;
}

function extractText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === 'string') return part;
        if (part && typeof part === 'object' && 'text' in part) {
          return String((part as { text: unknown }).text ?? '');
        }
        return '';
      })
      .filter(Boolean)
      .join('\n');
  }
  return '';
}

export function parseDevinSessionFile(dbPath: string, sessionId: string): Conversation {
  const empty: Conversation = {
    sessionId,
    project: '',
    messages: [],
    startTime: new Date(),
    endTime: new Date(),
    processedBytes: 0,
    sourceType: 'devin',
  };

  const db: ReadonlyDb | null = openReadonly(dbPath);
  if (!db) return empty;

  try {
    const sessionRow = db
      .prepare(`SELECT working_directory FROM sessions WHERE id = ?`)
      .get(sessionId) as { working_directory?: string } | undefined;
    const cwd = sessionRow?.working_directory ?? '';
    empty.project = getDevinProjectName(cwd) || 'devin';

    // message_nodes is a forest: every turn commits a new chain that replays
    // the earlier context under fresh node ids — but replays keep the SAME
    // message_id, so dedupe on it (first occurrence = original chain).
    // A long session holds ~800MB of replayed JSON; a regex prescan picks
    // role/message_id out of the text so we only JSON.parse each logical
    // message once (~12x faster than parsing every row).
    // ORDER BY row_id (the rowid alias): a free index traversal — sorting by
    // created_at would spill the payloads to temp storage and die with
    // SQLITE_FULL on a tight disk.
    const rows = db
      .prepare(
        `SELECT row_id, node_id, chat_message, created_at
         FROM message_nodes WHERE session_id = ?
         ORDER BY row_id ASC`
      )
      .all(sessionId) as unknown as Array<{ row_id: number; node_id: number; chat_message: string; created_at: number }>;

    const seen = new Set<string>();
    const messages: Message[] = [];
    for (const row of rows) {
      const role = ROLE_RE.exec(row.chat_message)?.[1];
      if (role !== 'user' && role !== 'assistant') continue;

      const uuid = ID_RE.exec(row.chat_message)?.[1] ?? `${sessionId}-${row.node_id}`;
      if (seen.has(uuid)) continue;

      let parsed: DevinChatMessage;
      try {
        parsed = JSON.parse(row.chat_message);
      } catch {
        continue;
      }

      const content = extractText(parsed.content).trim();
      if (!content) continue;
      seen.add(uuid);

      const timestamp = sqliteTs(row.created_at, new Date());
      messages.push({
        role,
        content,
        timestamp,
        source: {
          sessionId,
          messageUuid: uuid,
          timestamp,
          filePath: dbPath,
        },
      });
    }

    empty.messages = messages;
    if (messages.length) {
      empty.startTime = messages[0].timestamp;
      empty.endTime = messages[messages.length - 1].timestamp;
    }
    try { empty.processedBytes = statSync(dbPath).size; } catch {}
    return empty;
  } catch {
    return empty;
  } finally {
    try { db.close(); } catch {}
  }
}
