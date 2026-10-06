/**
 * Perplexity session parser — reads the Perplexity "personal computer" app's
 * local store at ~/.pplx/users/<user>/local.db.
 *
 * local.db is a SQLCipher v4 database (AES-256-CBC pages, HMAC-SHA512,
 * raw 256-bit key). The key is stored in Windows Credential Manager by the
 * app as a generic credential named
 *   local-mode-sqlcipher-key-<id>.ai.perplexity.desktop
 * (keyring-rs target layout: "<entry>.<service>"). We read it at runtime via
 * a PowerShell CredEnumerateW/CredReadW call — never persisted, never logged.
 *
 * Read path: decrypt each 4KiB page ([ciphertext][iv:16][hmac:64]) plus each
 * WAL frame's page payload, overlay WAL onto the main file, write a plaintext
 * SQLite image to a temp file, and open it with node:sqlite. HMACs are not
 * verified (read-only use).
 *
 * Schema (observed, Perplexity Computer 2026.9):
 *   sessions(id, user_id, title, model, status, created_at, updated_at, ...)
 *   trajectories(session_id, messages JSON [{role, content}], loaded_skills, updated_at)
 *   events(id, session_id, run_id, seq, payload JSON, created_at)
 *   files / grants / automations / automation_runs
 *
 * Also exposes getPerplexityInferenceConfig(): the app's local-mode inference
 * endpoint (LM Studio-style OpenAI-compatible server) which prose reuses as
 * an LLM fallback provider.
 */

import { readFileSync, readdirSync, existsSync, statSync, writeFileSync, mkdirSync } from 'fs';
import { join, basename, dirname } from 'path';
import { homedir, tmpdir } from 'os';
import { execFileSync } from 'child_process';
import { createDecipheriv, createHash } from 'crypto';
import type { Conversation, Message, SessionFile } from './session-parser.js';
import { openReadonly } from './sqlite.js';

// ============================================================================
// Data dir discovery
// ============================================================================

/**
 * Candidate roots for the Perplexity app's Electron profile directory.
 * Windows MSIX: %LOCALAPPDATA%/Packages/PerplexityAI.PerplexityApp_&lt;suffix&gt;/
 *   LocalCache/Roaming/Perplexity
 * Unpackaged Electron fallback: %APPDATA%/Perplexity (win),
 *   ~/Library/Application Support/Perplexity (mac), ~/.config/Perplexity.
 */
export function getPerplexityDataDirs(): string[] {
  const home = homedir();
  const dirs: string[] = [];

  if (process.platform === 'win32') {
    const local = process.env.LOCALAPPDATA || join(home, 'AppData', 'Local');
    const packages = join(local, 'Packages');
    try {
      for (const entry of readdirSync(packages, { withFileTypes: true })) {
        if (!entry.isDirectory() || !/^PerplexityAI\./i.test(entry.name)) continue;
        const candidate = join(packages, entry.name, 'LocalCache', 'Roaming', 'Perplexity');
        if (existsSync(candidate)) dirs.push(candidate);
      }
    } catch {}
    const appData = process.env.APPDATA || join(home, 'AppData', 'Roaming');
    dirs.push(join(appData, 'Perplexity'));
  } else if (process.platform === 'darwin') {
    dirs.push(join(home, 'Library', 'Application Support', 'Perplexity'));
  } else {
    const config = process.env.XDG_CONFIG_HOME || join(home, '.config');
    dirs.push(join(config, 'Perplexity'));
  }

  return dirs.filter((d) => existsSync(d));
}

/**
 * ~/.pplx/users/<id>/local.db paths — the agent's real conversation store.
 */
export function getPplxLocalDbPaths(): string[] {
  const root = join(homedir(), '.pplx', 'users');
  const out: string[] = [];
  try {
    if (!existsSync(root)) return out;
    for (const entry of readdirSync(root, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const p = join(root, entry.name, 'local.db');
      if (existsSync(p)) out.push(p);
    }
  } catch {}
  return out;
}

// ============================================================================
// Inference config (used by the LLM layer as a fallback provider)
// ============================================================================

export interface PerplexityInferenceConfig {
  baseUrl: string;
  model?: string;
  apiKey?: string;
}

/**
 * Read the app's local inference settings (user-settings.json →
 * inference_base_url / inference_model / inference_api_key). Returns null
 * when unset or not a local endpoint.
 */
export function getPerplexityInferenceConfig(): PerplexityInferenceConfig | null {
  for (const dir of getPerplexityDataDirs()) {
    const p = join(dir, 'user-settings.json');
    if (!existsSync(p)) continue;
    try {
      const values = JSON.parse(readFileSync(p, 'utf-8'))?.values;
      const baseUrl: unknown = values?.inference_base_url;
      if (typeof baseUrl !== 'string' || !baseUrl) continue;
      const cfg: PerplexityInferenceConfig = { baseUrl };
      if (typeof values.inference_model === 'string' && values.inference_model) {
        cfg.model = values.inference_model;
      }
      if (typeof values.inference_api_key === 'string' && values.inference_api_key) {
        cfg.apiKey = values.inference_api_key;
      }
      return cfg;
    } catch {
      continue;
    }
  }
  return null;
}

// ============================================================================
// SQLCipher v4 read-only decryption
// ============================================================================

const CIPHER_PAGE = 4096;
const CIPHER_RESERVE = 80; // iv(16) + hmac(64)
const SQLITE_MAGIC = Buffer.from('SQLite format 3\0', 'latin1');

let cachedKeyHex: string | null | undefined;

/**
 * Fetch the app's sqlcipher key from Windows Credential Manager.
 * The app stores it via keyring-rs as a generic credential whose target is
 * "<entry>.<service>": "local-mode-sqlcipher-key-<hex>.ai.perplexity.desktop".
 * Returns the 64-char hex string, or null when unavailable.
 */
function getPplxSqlcipherKeyHex(): string | null {
  if (cachedKeyHex !== undefined) return cachedKeyHex;
  cachedKeyHex = null;
  if (process.platform !== 'win32') return cachedKeyHex;

  const script = `
$ProgressPreference = 'SilentlyContinue'
$code = @'
using System;
using System.Runtime.InteropServices;
public static class PplxCred {
  [DllImport("advapi32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
  static extern bool CredEnumerateW(string filter, int flags, out int count, out IntPtr creds);
  [DllImport("advapi32.dll")] static extern void CredFree(IntPtr buffer);
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
  class CREDENTIAL {
    public int Flags; public int Type; public string TargetName; public string Comment;
    public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
    public int CredentialBlobSize; public IntPtr CredentialBlob;
    public int Persist; public int AttributeCount; public IntPtr Attributes;
    public string TargetAlias; public string UserName;
  }
  public static string Get(string prefix) {
    int n; IntPtr arr;
    if (!CredEnumerateW(null, 0, out n, out arr)) return "";
    try {
      for (int i = 0; i < n; i++) {
        IntPtr p = Marshal.ReadIntPtr(arr, i * IntPtr.Size);
        CREDENTIAL c = (CREDENTIAL)Marshal.PtrToStructure(p, typeof(CREDENTIAL));
        if (c.TargetName != null && c.TargetName.StartsWith(prefix) && c.CredentialBlob != IntPtr.Zero) {
          byte[] b = new byte[c.CredentialBlobSize];
          Marshal.Copy(c.CredentialBlob, b, 0, b.Length);
          return System.Text.Encoding.Unicode.GetString(b);
        }
      }
      return "";
    } finally { CredFree(arr); }
  }
}
'@
Add-Type -TypeDefinition $code
[PplxCred]::Get('local-mode-sqlcipher-key-')
`;

  try {
    const encoded = Buffer.from(script, 'utf16le').toString('base64');
    const out = execFileSync(
      'powershell.exe',
      ['-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
      { encoding: 'utf-8', timeout: 20_000 }
    ).trim();
    const m = out.match(/[0-9a-fA-F]{64}/);
    if (m) cachedKeyHex = m[0];
  } catch {}
  return cachedKeyHex;
}

/**
 * Decrypt one 4KiB SQLCipher page → plaintext page (with 80B zero reserve).
 * Page 1 stores the 16-byte salt in place of the SQLite header magic; the
 * magic is restored after decryption.
 */
function decryptPage(page: Buffer, key: Buffer, isFirst: boolean): Buffer {
  const ct = isFirst
    ? page.subarray(16, page.length - CIPHER_RESERVE)
    : page.subarray(0, page.length - CIPHER_RESERVE);
  const iv = page.subarray(page.length - CIPHER_RESERVE, page.length - CIPHER_RESERVE + 16);
  const d = createDecipheriv('aes-256-cbc', key, iv);
  d.setAutoPadding(false);
  const pt = Buffer.concat([d.update(ct), d.final()]);
  const body = isFirst ? Buffer.concat([SQLITE_MAGIC, pt]) : pt;
  return Buffer.concat([body, Buffer.alloc(CIPHER_RESERVE)]);
}

interface DecryptedDb {
  plainPath: string;
  stamp: string;
}

const decryptCache = new Map<string, DecryptedDb>();

/**
 * Decrypt a SQLCipher local.db (+WAL) into a plaintext sqlite temp file.
 * Cached per (path, db mtime+size, wal size): repeated parses within and
 * across prose invocations reuse the image until the source changes.
 */
function decryptedDbPath(dbPath: string): string | null {
  const keyHex = getPplxSqlcipherKeyHex();
  if (!keyHex) return null;

  let dbStat;
  try { dbStat = statSync(dbPath); } catch { return null; }
  const walPath = `${dbPath}-wal`;
  const walSize = existsSync(walPath) ? statSync(walPath).size : 0;
  const stamp = `${dbStat.mtimeMs}-${dbStat.size}-${walSize}`;

  const hit = decryptCache.get(dbPath);
  if (hit && hit.stamp === stamp && existsSync(hit.plainPath)) return hit.plainPath;

  let plainPath: string;
  try {
    const key = Buffer.from(keyHex, 'hex');
    const dbBuf = readFileSync(dbPath);
    const walBuf = existsSync(walPath) ? readFileSync(walPath) : null;

    const pages = new Map<number, Buffer>();
    for (let off = 0; off + CIPHER_PAGE <= dbBuf.length; off += CIPHER_PAGE) {
      const pgno = off / CIPHER_PAGE + 1;
      pages.set(pgno, decryptPage(dbBuf.subarray(off, off + CIPHER_PAGE), key, pgno === 1));
    }

    // Overlay WAL frames (each frame's page payload is encrypted identically).
    // Last write per page wins; the commit frame's dbsize fixes the page count.
    let maxPage = pages.size;
    let dbSize = 0;
    if (walBuf && walBuf.length >= 32) {
      const frameSize = 24 + CIPHER_PAGE;
      const frames = Math.floor((walBuf.length - 32) / frameSize);
      for (let f = 0; f < frames; f++) {
        const off = 32 + f * frameSize;
        const pgno = walBuf.readUInt32BE(off);
        const commitDbSize = walBuf.readUInt32BE(off + 4);
        if (pgno < 1 || pgno > 0x7fffffff) break; // torn tail
        pages.set(pgno, decryptPage(walBuf.subarray(off + 24, off + frameSize), key, pgno === 1));
        if (commitDbSize) dbSize = commitDbSize;
        if (pgno > maxPage) maxPage = pgno;
      }
    }

    const total = Math.max(dbSize, maxPage, 1);
    const tag = createHash('sha1')
      .update(dbPath).update('\0').update(stamp)
      .digest('hex').slice(0, 12);
    const dir = join(tmpdir(), 'prose-pplx');
    mkdirSync(dir, { recursive: true });
    plainPath = join(dir, `plain-${tag}.db`);

    const out = Buffer.alloc(total * CIPHER_PAGE);
    for (let i = 1; i <= total; i++) {
      const page = pages.get(i);
      if (page) page.copy(out, (i - 1) * CIPHER_PAGE);
    }
    writeFileSync(plainPath, out);
  } catch {
    return null;
  }

  decryptCache.set(dbPath, { plainPath, stamp });
  return plainPath;
}

// ============================================================================
// local.db schema access
// ============================================================================

interface PplxSessionRow {
  id: string;
  title: string | null;
  model: string | null;
  status: string | null;
  created_at: string;
  updated_at: string;
}

function pplxTs(iso: string | null | undefined, fallback: Date): Date {
  if (!iso) return fallback;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? new Date(t) : fallback;
}

function pplxWorkspaceFor(dbPath: string, sessionId: string): string | undefined {
  // local.db sits at <user>/local.db; workspaces live at <user>/workspaces/<id>
  const ws = join(dirname(dbPath), 'workspaces', sessionId);
  return existsSync(ws) ? ws : undefined;
}

/**
 * Open the decrypted image of a local.db. Returns null when the key or the
 * database is unavailable. Caller must close.
 */
function openPplxDb(dbPath: string) {
  const plain = decryptedDbPath(dbPath);
  if (!plain) return null;
  return openReadonly(plain);
}

// ============================================================================
// Discovery & parse
// ============================================================================

export function discoverPerplexitySessionFiles(): SessionFile[] {
  const out: SessionFile[] = [];
  for (const dbPath of getPplxLocalDbPaths()) {
    const db = openPplxDb(dbPath);
    if (!db) continue;
    try {
      const fileSize = statSync(dbPath).size;
      const rows = db
        .prepare(`SELECT id, title, model, status, created_at, updated_at FROM sessions`)
        .all() as unknown as PplxSessionRow[];
      for (const row of rows) {
        out.push({
          path: dbPath,
          sessionId: `pplx-${row.id}`,
          project: 'perplexity',
          modifiedTime: pplxTs(row.updated_at, statSync(dbPath).mtime),
          fileSize,
          sourceType: 'perplexity',
          cwd: pplxWorkspaceFor(dbPath, row.id),
        });
      }
    } catch {
      // schema drift or partial decrypt — skip this db
    } finally {
      try { db.close(); } catch {}
    }
  }
  return out.sort((a, b) => b.modifiedTime.getTime() - a.modifiedTime.getTime());
}

interface PplxTrajectoryRow {
  session_id: string;
  messages: string;
}

export function parsePerplexitySessionFile(filePath: string, sessionId?: string): Conversation {
  const id = sessionId ?? `pplx-${basename(filePath)}`;
  const empty: Conversation = {
    sessionId: id,
    project: 'perplexity',
    messages: [],
    startTime: new Date(),
    endTime: new Date(),
    processedBytes: 0,
    sourceType: 'perplexity',
  };
  if (!filePath.endsWith('.db')) return empty;

  const db = openPplxDb(filePath);
  if (!db) return empty;

  try {
    const rawId = id.replace(/^pplx-/, '');
    const sess = db
      .prepare(`SELECT id, title, model, status, created_at, updated_at FROM sessions WHERE id = ?`)
      .get(rawId) as PplxSessionRow | undefined;
    const mtime = statSync(filePath).mtime;

    const traj = db
      .prepare(`SELECT session_id, messages FROM trajectories WHERE session_id = ?`)
      .get(rawId) as PplxTrajectoryRow | undefined;

    let rawMessages: Array<{ role?: string; content?: unknown }> = [];
    if (traj?.messages) {
      try {
        const parsed = JSON.parse(traj.messages);
        if (Array.isArray(parsed)) rawMessages = parsed;
      } catch {}
    }

    const startTs = pplxTs(sess?.created_at, mtime);
    const endTs = pplxTs(sess?.updated_at, mtime);

    // Trajectory messages carry no timestamps — spread them linearly across
    // the session window so stats/grep ordering stays meaningful.
    const kept = rawMessages.filter(
      (m) => (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string' && (m.content as string).trim()
    );
    const span = Math.max(0, endTs.getTime() - startTs.getTime());
    const messages: Message[] = kept.map((m, i) => {
      const timestamp = new Date(startTs.getTime() + (kept.length > 1 ? (span * i) / (kept.length - 1) : 0));
      return {
        role: m.role as 'user' | 'assistant',
        content: (m.content as string).trim(),
        timestamp,
        source: {
          sessionId: id,
          messageUuid: `${id}-${i}`,
          timestamp,
          filePath,
        },
      };
    });

    empty.project = 'perplexity';
    empty.messages = messages;
    empty.startTime = messages.length ? messages[0].timestamp : startTs;
    empty.endTime = messages.length ? messages[messages.length - 1].timestamp : endTs;
    try { empty.processedBytes = statSync(filePath).size; } catch {}
    return empty;
  } catch {
    return empty;
  } finally {
    try { db.close(); } catch {}
  }
}
