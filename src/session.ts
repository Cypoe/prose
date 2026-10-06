/**
 * Session — verbatim readout of a single session by id (or id prefix).
 *
 * Scans all known session files (Claude Code CLI, ACP, Codex, opencode) and
 * resolves `<id>` against `sessionId`, accepting any unique prefix. Renders
 * the matched session verbatim in the same header format as `snap`, with
 * optional tail/since slicing. No cwd filter, no live-session skipping —
 * if you named the id, you want it.
 */

import {
  discoverSessionFiles,
  parseSessionFile,
  type Conversation,
  type Message,
  type SessionFile,
} from './session-parser.js';
import {
  discoverCodexSessionFiles,
  parseCodexSessionFile,
} from './codex-session-parser.js';
import {
  discoverOpencodeSessionFiles,
  parseOpencodeSessionFile,
} from './opencode-session-parser.js';
import {
  discoverCursorSessionFiles,
  parseCursorSessionFile,
} from './cursor-session-parser.js';
import {
  discoverPiSessionFiles,
  discoverOmpSessionFiles,
  parsePiSessionFile,
} from './pi-session-parser.js';
import {
  getAntigravityBrains,
  getAntigravityArtifacts,
  parseAntigravityArtifact,
} from './source-parsers.js';
import {
  discoverDevinSessionFiles,
  parseDevinSessionFile,
} from './devin-session-parser.js';
import {
  discoverPerplexitySessionFiles,
  parsePerplexitySessionFile,
} from './perplexity-session-parser.js';

export interface SessionOptions {
  /** Last N messages of the session (default: all). */
  turns?: number;
  /** Only include messages at or after this timestamp. */
  since?: Date;
  /**
   * Per-message byte cap. 0 (default) disables truncation — when you ask for
   * one specific session, you almost always want the full text.
   */
  maxMessageBytes?: number;
  /**
   * Include `entrypoint: 'sdk-cli'` Claude Code sessions in the candidate
   * pool. Default true here (unlike snap/whisper): the user named an id
   * explicitly, so we shouldn't silently refuse to find it.
   */
  includeSdkCli?: boolean;
}

export interface SessionMatch {
  sessionId: string;
  sourceType: NonNullable<SessionFile['sourceType']>;
  path: string;
  modifiedTime: string;
  cwd?: string;
}

export interface SessionResult {
  sessionId: string;
  sourceType: NonNullable<SessionFile['sourceType']>;
  path: string;
  cwd?: string;
  startTime: string;
  endTime: string;
  modifiedTime: string;
  ageMs: number;
  messageCount: number;
  /** How many messages were rendered after `turns`/`since` filters. */
  messagesIncluded: number;
  bytes: number;
  text: string;
}

export class SessionNotFoundError extends Error {
  constructor(public readonly id: string) {
    super(`No session matches id prefix '${id}'.`);
    this.name = 'SessionNotFoundError';
  }
}

export class SessionAmbiguousError extends Error {
  constructor(public readonly id: string, public readonly matches: SessionMatch[]) {
    super(
      `Ambiguous id prefix '${id}' — matched ${matches.length} sessions:\n` +
        matches.map((m) => `  ${m.sessionId}  ${m.sourceType}  ${m.modifiedTime}  ${m.cwd ?? ''}`).join('\n')
    );
    this.name = 'SessionAmbiguousError';
  }
}

function formatAge(ms: number): string {
  if (ms < 60_000) return `${Math.round(ms / 1000)}s ago`;
  if (ms < 3_600_000) return `${Math.round(ms / 60_000)}m ago`;
  if (ms < 86_400_000) return `${Math.round(ms / 3_600_000)}h ago`;
  return `${Math.round(ms / 86_400_000)}d ago`;
}

function clipMessageContent(content: string, maxBytes: number): string {
  if (maxBytes <= 0) return content;
  const buf = Buffer.from(content, 'utf-8');
  if (buf.length <= maxBytes) return content;
  const head = buf.subarray(0, maxBytes).toString('utf-8');
  const elided = buf.length - maxBytes;
  return `${head}\n... [${elided} bytes elided]`;
}

function sourceLabel(t: NonNullable<SessionFile['sourceType']>): string {
  switch (t) {
    case 'codex':
      return 'Codex';
    case 'opencode':
      return 'opencode';
    case 'claude-code':
      return 'Claude Code';
    case 'antigravity':
      return 'Antigravity';
    case 'cursor':
      return 'Cursor';
    case 'pi':
      return 'pi';
    case 'omp':
      return 'OMP';
    case 'devin':
      return 'Devin';
    case 'perplexity':
      return 'Perplexity';
    default:
      return t;
  }
}

export function collectAllSessionFiles(): SessionFile[] {
  // No projectPath filter on any source — we want the full id-space.
  const claude = discoverSessionFiles();
  const codex = discoverCodexSessionFiles();
  const opencode = discoverOpencodeSessionFiles();
  const cursor = discoverCursorSessionFiles();
  const pi = discoverPiSessionFiles();
  const omp = discoverOmpSessionFiles();
  const devin = discoverDevinSessionFiles();
  const perplexity = discoverPerplexitySessionFiles();

  const antigravity: SessionFile[] = [];
  try {
    const brains = getAntigravityBrains();
    for (const brain of brains) {
      const artifacts = getAntigravityArtifacts(brain, '');
      for (const art of artifacts) {
        antigravity.push({ ...art, sourceType: 'antigravity' });
      }
    }
  } catch {
    // Ignore errors
  }

  return [...claude, ...codex, ...opencode, ...cursor, ...pi, ...omp, ...devin, ...perplexity, ...antigravity];
}

export function parseByType(f: SessionFile): Conversation {
  if (f.sourceType === 'codex') return parseCodexSessionFile(f.path);
  if (f.sourceType === 'opencode') return parseOpencodeSessionFile(f.path);
  if (f.sourceType === 'cursor') return parseCursorSessionFile(f.path);
  if (f.sourceType === 'pi' || f.sourceType === 'omp') return parsePiSessionFile(f.path, f.sourceType);
  if (f.sourceType === 'devin') return parseDevinSessionFile(f.path, f.sessionId);
  if (f.sourceType === 'perplexity') return parsePerplexitySessionFile(f.path, f.sessionId);
  if (f.sourceType === 'antigravity') {
    const messages = parseAntigravityArtifact(f.path, f.sessionId, f.project);
    const startTime = messages.length ? messages[0].timestamp : f.modifiedTime;
    const endTime = messages.length ? messages[messages.length - 1].timestamp : f.modifiedTime;
    return {
      sessionId: f.sessionId,
      project: f.project,
      messages,
      startTime,
      endTime,
      processedBytes: f.fileSize,
      sourceType: 'antigravity',
    };
  }
  return parseSessionFile(f.path);
}

/**
 * Resolve a session id prefix to its matching files. Exposed for callers that
 * want to handle the 0/many cases themselves (e.g. shell completion).
 */
export function resolveSessionId(idPrefix: string): SessionFile[] {
  if (!idPrefix) return [];
  const all = collectAllSessionFiles();
  return all.filter((f) => f.sessionId.startsWith(idPrefix));
}

export function session(idPrefix: string, opts: SessionOptions = {}): SessionResult {
  const maxMessageBytes = opts.maxMessageBytes ?? 0;
  const includeSdkCli = opts.includeSdkCli ?? true;

  const matches = resolveSessionId(idPrefix);
  if (matches.length === 0) throw new SessionNotFoundError(idPrefix);
  if (matches.length > 1) {
    // Sort newest-first to make the error easier to read.
    const sorted = [...matches].sort(
      (a, b) => b.modifiedTime.getTime() - a.modifiedTime.getTime()
    );
    throw new SessionAmbiguousError(
      idPrefix,
      sorted.map((m) => ({
        sessionId: m.sessionId,
        sourceType: m.sourceType ?? 'claude-code',
        path: m.path,
        modifiedTime: m.modifiedTime.toISOString(),
        cwd: m.cwd,
      }))
    );
  }

  const file = matches[0];
  const conv = parseByType(file);

  if (!includeSdkCli && conv.entrypoint === 'sdk-cli') {
    // We matched a session whose id was unique but is a Claude-Code automation
    // run the caller asked to exclude. Surface this as not-found rather than
    // silently emitting an empty render.
    throw new SessionNotFoundError(idPrefix);
  }

  let messages: Message[] = conv.messages;
  if (opts.since) {
    const sinceMs = opts.since.getTime();
    messages = messages.filter((m) => m.timestamp.getTime() >= sinceMs);
  }
  if (typeof opts.turns === 'number' && opts.turns > 0) {
    messages = messages.slice(-opts.turns);
  }

  const sType = file.sourceType ?? 'claude-code';
  const lastMessageTime = conv.messages.length
    ? conv.messages[conv.messages.length - 1].timestamp
    : file.modifiedTime;
  const ageMs = Date.now() - lastMessageTime.getTime();

  const lines: string[] = [];
  lines.push(
    `=== ${sourceLabel(sType)} session ${conv.sessionId.slice(0, 8)} (${formatAge(ageMs)}, last message ${lastMessageTime.toISOString()}) ===`
  );
  lines.push(`# full id:  ${conv.sessionId}`);
  lines.push(`# path:     ${file.path}`);
  if (file.cwd) lines.push(`# cwd:      ${file.cwd}`);
  if (conv.entrypoint) lines.push(`# entry:    ${conv.entrypoint}`);
  lines.push(
    `# messages: ${messages.length} of ${conv.messages.length}` +
      (opts.turns ? ` (tail ${opts.turns})` : '') +
      (opts.since ? ` since ${opts.since.toISOString()}` : '')
  );
  lines.push('');
  for (const msg of messages) {
    lines.push(`[${msg.timestamp.toISOString()}] ${msg.role.toUpperCase()}:`);
    lines.push(clipMessageContent(msg.content, maxMessageBytes));
    lines.push('');
  }
  const text = lines.join('\n');

  return {
    sessionId: conv.sessionId,
    sourceType: sType,
    path: file.path,
    cwd: file.cwd,
    startTime: conv.startTime.toISOString(),
    endTime: conv.endTime.toISOString(),
    modifiedTime: file.modifiedTime.toISOString(),
    ageMs,
    messageCount: conv.messages.length,
    messagesIncluded: messages.length,
    bytes: Buffer.byteLength(text, 'utf-8'),
    text,
  };
}
