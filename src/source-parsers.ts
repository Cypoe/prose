/**
 * Source Parsers - Ingest data from external sources (Git, Antigravity)
 */

import { execSync } from 'child_process';
import { readFileSync, readdirSync, existsSync, statSync, openSync, readSync, closeSync } from 'fs';
import { join, basename } from 'path';
import { homedir } from 'os';
import type { Message, SessionFile } from './session-parser.js';
import { openReadonly } from './sqlite.js';

// ============================================================================
// Git Integration
// ============================================================================

/**
 * Get the latest git commits as "messages"
 */
export function getGitCommits(repoPath: string, limit: number = 50): Message[] {
    try {
        const output = execSync(
            `git -C "${repoPath}" log -n ${limit} --pretty=format:"commit|%H|%at|%an|%s%n%b%n---"`,
            { encoding: 'utf-8' }
        );

        const commits = output.split('\n---\n').filter(Boolean);
        const messages: Message[] = [];

        for (const commit of commits) {
            const lines = commit.split('\n');
            const header = lines[0].split('|');
            if (header[0] !== 'commit') continue;

            const hash = header[1];
            const timestamp = new Date(parseInt(header[2], 10) * 1000);
            const author = header[3];
            const subject = header[4];
            const body = lines.slice(1).join('\n').trim();

            messages.push({
                role: 'user', // Commits are user intent
                content: `GIT COMMIT: ${subject}\n\n${body}`,
                timestamp,
                source: {
                    sessionId: `git-${hash.slice(0, 8)}`,
                    messageUuid: hash,
                    timestamp,
                    filePath: repoPath,
                },
            });
        }

        return messages;
    } catch (error) {
        return [];
    }
}

/**
 * Check if a directory is a git repository
 */
export function isGitRepo(path: string): boolean {
    try {
        execSync(`git -C "${path}" rev-parse --is-inside-work-tree`, { stdio: 'ignore' });
        return true;
    } catch {
        return false;
    }
}

// ============================================================================
// Antigravity Integration
// ============================================================================

/**
 * Discover Antigravity brain directories
 */
export function getAntigravityBrains(): string[] {
    const metaDir = join(homedir(), '.gemini', 'antigravity', 'brain');
    if (!existsSync(metaDir)) return [];

    try {
        return readdirSync(metaDir, { withFileTypes: true })
            .filter(d => d.isDirectory())
            .map(d => join(metaDir, d.name));
    } catch {
        return [];
    }
}

/**
 * Directory holding per-conversation stores: <uuid>.db (SQLite trajectory,
 * current IDE) and <uuid>.pb (protobuf, legacy).
 */
export function getAntigravityConversationsDir(): string {
    return join(homedir(), '.gemini', 'antigravity', 'conversations');
}

/**
 * Discover and parse Antigravity artifacts and transcripts as pseudo-sessions
 */
export function getAntigravityArtifacts(brainPath: string, projectName: string): SessionFile[] {
    if (!existsSync(brainPath)) return [];

    const artifacts: SessionFile[] = [];

    // 1. Discover transcript.jsonl if present
    const transcriptPath = join(brainPath, '.system_generated', 'logs', 'transcript.jsonl');
    const uuid = basename(brainPath);
    if (existsSync(transcriptPath)) {
        const stats = statSync(transcriptPath);
        artifacts.push({
            path: transcriptPath,
            sessionId: `anti-${uuid}-transcript`,
            project: projectName,
            modifiedTime: stats.mtime,
            fileSize: stats.size,
        });
    } else {
        // No transcript — fall back to the conversation store. The .db
        // trajectory (newer IDE) and .pb dump (legacy) both carry protobuf
        // step payloads we can text-extract. Prefer .db when both exist.
        const convDir = getAntigravityConversationsDir();
        for (const ext of ['db', 'pb']) {
            const convPath = join(convDir, `${uuid}.${ext}`);
            if (existsSync(convPath)) {
                const stats = statSync(convPath);
                artifacts.push({
                    path: convPath,
                    sessionId: `anti-${uuid}-conversation`,
                    project: projectName,
                    modifiedTime: stats.mtime,
                    fileSize: stats.size,
                });
                break;
            }
        }
    }

    // 2. Discover standard markdown artifacts (.md files)
    const files = readdirSync(brainPath, { withFileTypes: true })
        .filter(f => f.isFile() && f.name.endsWith('.md'));

    for (const file of files) {
        const filePath = join(brainPath, file.name);
        const stats = statSync(filePath);
        const sessionId = `anti-${basename(brainPath)}-${file.name.replace('.md', '')}`;

        artifacts.push({
            path: filePath,
            sessionId,
            project: projectName,
            modifiedTime: stats.mtime,
            fileSize: stats.size,
        });
    }

    return artifacts;
}

/**
 * Read and format an Antigravity artifact or transcript as messages
 */
export function parseAntigravityArtifact(filePath: string, sessionId: string, project: string): Message[] {
    try {
        const stats = statSync(filePath);

        // 1. Handle transcript.jsonl
        if (filePath.endsWith('transcript.jsonl')) {
            const content = readFileSync(filePath, 'utf-8');
            const messages: Message[] = [];

            for (const line of content.split('\n')) {
                if (!line.trim()) continue;
                try {
                    const parsed = JSON.parse(line);
                    const timestamp = new Date(parsed.created_at || parsed.timestamp || stats.mtime);

                    if (parsed.type === 'USER_INPUT' || parsed.source === 'USER_EXPLICIT') {
                        if (parsed.content && typeof parsed.content === 'string') {
                            messages.push({
                                role: 'user',
                                content: parsed.content,
                                timestamp,
                                source: {
                                    sessionId,
                                    messageUuid: `${sessionId}-${parsed.step_index ?? messages.length}`,
                                    timestamp,
                                    filePath,
                                }
                            });
                        }
                    } else if (parsed.source === 'MODEL') {
                        if (parsed.content && typeof parsed.content === 'string' && parsed.content.trim()) {
                            messages.push({
                                role: 'assistant',
                                content: parsed.content,
                                timestamp,
                                source: {
                                    sessionId,
                                    messageUuid: `${sessionId}-${parsed.step_index ?? messages.length}`,
                                    timestamp,
                                    filePath,
                                }
                            });
                        }
                    }
                } catch {
                    // Ignore line-parse errors (partial lines at end of active files)
                }
            }
            return messages;
        }

        // 2. Handle trajectory stores (.db SQLite / .pb protobuf)
        if (filePath.endsWith('.db')) {
            return parseAntigravityTrajectoryDb(filePath, sessionId);
        }
        if (filePath.endsWith('.pb')) {
            return parseAntigravityPbConversation(filePath, sessionId);
        }

        // 3. Handle standard markdown artifacts
        const content = readFileSync(filePath, 'utf-8');
        return [{
            role: 'assistant', // Artifacts are agent output
            content: `ANTIGRAVITY ARTIFACT:\n\n${content}`,
            timestamp: stats.mtime,
            source: {
                sessionId,
                messageUuid: sessionId, // Use session ID as UUID for artifacts
                timestamp: stats.mtime,
                filePath,
            },
        }];
    } catch {
        return [];
    }
}

/**
 * Generic protobuf text extractor. Walks a buffer treating it as a protobuf
 * message; recurses into length-delimited fields that don't decode as text.
 * Best-effort by design — the Cascade trajectory schema isn't published, so
 * we recover human-language strings, not structured fields.
 */
function extractProtoStrings(
    buf: Buffer | Uint8Array,
    out: string[] = [],
    depth: number = 0,
    cap: number = 400
): string[] {
    let i = 0;
    while (i < buf.length && out.length < cap) {
        // key varint
        let key = 0, shift = 0;
        while (i < buf.length) {
            const x = buf[i++];
            key |= (x & 0x7f) << shift;
            shift += 7;
            if (!(x & 0x80)) break;
        }
        const wire = key & 7;
        if (wire === 0) {
            while (i < buf.length && (buf[i] & 0x80)) i++;
            i++;
        } else if (wire === 5) {
            i += 4;
        } else if (wire === 1) {
            i += 8;
        } else if (wire === 2) {
            let n = 0; shift = 0;
            while (i < buf.length) {
                const x = buf[i++];
                n |= (x & 0x7f) << shift;
                shift += 7;
                if (!(x & 0x80)) break;
            }
            const seg = buf.subarray(i, i + n);
            i += n;
            if (n >= 16 && n <= 30000) {
                let printable = 0;
                const probe = Math.min(n, 400);
                for (let k = 0; k < probe; k++) {
                    const c = seg[k];
                    if ((c >= 32 && c < 127) || c === 9 || c === 10 || c === 13) printable++;
                }
                if (printable / probe > 0.85) {
                    try {
                        const t = Buffer.from(seg).toString('utf-8');
                        if (/[a-zA-Z]{4,}/.test(t)) {
                            out.push(t);
                            continue;
                        }
                    } catch { /* fall through to recursion */ }
                }
                if (depth < 4) extractProtoStrings(seg, out, depth + 1, cap);
            }
        } else {
            break; // wire types 3/4 (groups) and invalid → stop this level
        }
    }
    return out;
}

/**
 * Pick the most plausible human-language string from a step payload's
 * extracted strings: not JSON, not session/step IDs, not tool-call envelopes.
 */
function pickProseString(strings: string[]): string | null {
    let best: string | null = null;
    for (const s of strings) {
        const t = s.trim();
        if (t.length < 24) continue;
        if (t.startsWith('{') || t.startsWith('[')) continue;
        if (/sessionID|stepType|trajectory/i.test(t)) continue;
        if (/^[A-Za-z0-9_-]{15,}$/.test(t)) continue;
        if (!best || t.length > best.length) best = t;
    }
    return best;
}

/**
 * Parse a Cascade trajectory SQLite store (<uuid>.db).
 * Observed step_type semantics: 14 = user input, 15 = model text step
 * (often alongside a tool-call JSON blob — we keep the prose, not the JSON).
 */
function parseAntigravityTrajectoryDb(filePath: string, sessionId: string): Message[] {
    const db = openReadonly(filePath);
    if (!db) return [];
    try {
        const rows = db
            .prepare(`SELECT idx, step_type, step_payload FROM steps ORDER BY idx ASC`)
            .all() as unknown as Array<{ idx: number; step_type: number; step_payload: Uint8Array | null }>;

        const mtime = statSync(filePath).mtime;
        const messages: Message[] = [];
        const seen = new Set<string>();

        for (const row of rows) {
            if (!row.step_payload) continue;
            const role =
                row.step_type === 14 ? 'user'
                : row.step_type === 15 ? 'assistant'
                : null;
            if (!role) continue;

            const strings = extractProtoStrings(row.step_payload);
            const prose = pickProseString(strings);
            if (!prose || seen.has(prose)) continue;
            seen.add(prose);

            messages.push({
                role,
                content: prose,
                timestamp: mtime,
                source: {
                    sessionId,
                    messageUuid: `${sessionId}-step${row.idx}`,
                    timestamp: mtime,
                    filePath,
                },
            });
        }
        return messages;
    } catch {
        return [];
    } finally {
        try { db.close(); } catch {}
    }
}

/**
 * Parse a legacy <uuid>.pb conversation dump. Structure is unpublished;
 * we extract long prose strings and emit them as assistant pseudo-messages.
 */
function parseAntigravityPbConversation(filePath: string, sessionId: string): Message[] {
    try {
        const buf = readFileSync(filePath);
        const strings = extractProtoStrings(buf);
        const mtime = statSync(filePath).mtime;
        const messages: Message[] = [];
        const seen = new Set<string>();

        for (const s of strings) {
            const t = s.trim();
            if (t.length < 40 || seen.has(t)) continue;
            if (/^[A-Za-z0-9_-]{20,}$/.test(t)) continue;
            seen.add(t);
            messages.push({
                role: 'assistant',
                content: messages.length === 0
                    ? `ANTIGRAVITY CONVERSATION (protobuf extract):\n\n${t}`
                    : t,
                timestamp: mtime,
                source: {
                    sessionId,
                    messageUuid: `${sessionId}-${messages.length}`,
                    timestamp: mtime,
                    filePath,
                },
            });
        }
        return messages;
    } catch {
        return [];
    }
}

/**
 * Fuzzy match a brain to a project name by checking its task.md, implementation_plan.md, or transcript.jsonl
 */
export function matchBrainToProject(brainPath: string, projectFilter: string): boolean {
    const sanitizedFilter = projectFilter.replace(/^-Users-[^-]+-src-/, '').toLowerCase();

    // 1. Check task.md
    const taskPath = join(brainPath, 'task.md');
    if (existsSync(taskPath)) {
        try {
            const content = readFileSync(taskPath, 'utf-8').toLowerCase();
            if (content.includes(sanitizedFilter) ||
                sanitizedFilter.includes(basename(brainPath).toLowerCase())) {
                return true;
            }
        } catch {}
    }

    // 2. Check implementation_plan.md
    const planPath = join(brainPath, 'implementation_plan.md');
    if (existsSync(planPath)) {
        try {
            const content = readFileSync(planPath, 'utf-8').toLowerCase();
            if (content.includes(sanitizedFilter)) {
                return true;
            }
        } catch {}
    }

    // 3. Check transcript.jsonl (extremely robust!)
    const transcriptPath = join(brainPath, '.system_generated', 'logs', 'transcript.jsonl');
    if (existsSync(transcriptPath)) {
        try {
            const fd = openSync(transcriptPath, 'r');
            const buffer = Buffer.alloc(65536);
            const bytesRead = readSync(fd, buffer, 0, buffer.length, 0);
            closeSync(fd);
            const content = buffer.subarray(0, bytesRead).toString('utf-8').toLowerCase();

            if (content.includes(sanitizedFilter) ||
                content.includes(basename(projectFilter).toLowerCase()) ||
                projectFilter.toLowerCase().includes(basename(brainPath).toLowerCase())) {
                return true;
            }
        } catch {}
    }

    return false;
}

/**
 * Retrieval of latest git commit date for incremental git syncing
 */
export function getLatestGitCommitDate(repoPath: string): Date | null {
    try {
        const output = execSync(`git -C "${repoPath}" log -1 --format=%at`, {
            encoding: 'utf-8',
            stdio: ['ignore', 'pipe', 'ignore'],
        });
        const ts = parseInt(output.trim(), 10);
        if (!Number.isFinite(ts)) return null;
        return new Date(ts * 1000);
    } catch {
        return null;
    }
}

export interface GitCommitSummary {
    hash: string;
    subject: string;
    timestamp: Date;
    author: string;
}

/**
 * Time-windowed compact commit list for a repo. Used by standup to give the LLM
 * a ground-truth view of what shipped, alongside session tails that bias toward
 * end-of-session framings. Returns newest-first; empty array on any error.
 */
export function getCommitsSince(
    repoPath: string,
    since: Date,
    limit: number = 50
): GitCommitSummary[] {
    try {
        const output = execSync(
            `git -C "${repoPath}" log -n ${limit} --since="${since.toISOString()}" --pretty=format:"%H|%at|%an|%s"`,
            { encoding: 'utf-8' }
        );
        if (!output.trim()) return [];
        const commits: GitCommitSummary[] = [];
        for (const line of output.split('\n')) {
            const parts = line.split('|');
            if (parts.length < 4) continue;
            const [hash, at, author, ...rest] = parts;
            commits.push({
                hash,
                timestamp: new Date(parseInt(at, 10) * 1000),
                author,
                subject: rest.join('|'),
            });
        }
        return commits;
    } catch {
        return [];
    }
}

// ============================================================================
// Intelligent Design Integration
// ============================================================================

/**
 * Discover Intelligent Design sessions in .claude/prose/
 */
export function getDesignSessions(rootPath: string, projectName: string): SessionFile[] {
    const proseDir = join(rootPath, '.claude', 'prose');
    if (!existsSync(proseDir)) return [];

    const artifacts: SessionFile[] = [];
    try {
        const files = readdirSync(proseDir, { withFileTypes: true })
            .filter(f => f.isFile() && f.name.startsWith('design-') && f.name.endsWith('.json'));

        for (const file of files) {
            const filePath = join(proseDir, file.name);
            const stats = statSync(filePath);
            const sessionId = file.name.replace('.json', '');

            artifacts.push({
                path: filePath,
                sessionId,
                project: projectName,
                modifiedTime: stats.mtime,
                fileSize: stats.size,
            });
        }
    } catch {
        // Ignore errors
    }

    return artifacts;
}

/**
 * Parse a design session JSON file into messages
 */
export function parseDesignSession(filePath: string, sessionId: string): Message[] {
    try {
        const content = readFileSync(filePath, 'utf-8');
        const artifact = JSON.parse(content);
        const stats = statSync(filePath);

        if (artifact.type !== 'design-session') return [];

        return artifact.messages.map((m: any, i: number) => ({
            role: m.role,
            content: i === 0 ? `INTELLIGENT DESIGN SESSION:\n\n${m.content}` : m.content,
            timestamp: stats.mtime,
            source: {
                sessionId,
                messageUuid: `${sessionId}-${i}`,
                timestamp: stats.mtime,
                filePath,
            },
        }));
    } catch {
        return [];
    }
}
