/**
 * Memory Bank - Persistent storage for evolved fragments
 *
 * Stores fragments with:
 * - Temporal metadata (when evolved, from which sessions)
 * - Source links (provenance to exact messages)
 * - Searchable index for fractal-grep
 */

import { existsSync, readFileSync, writeFileSync, mkdirSync, renameSync } from 'fs';
import { join, dirname } from 'path';
import { homedir } from 'os';
import { execSync } from 'child_process';
import { createHash } from 'crypto';

import type { SourceLink, Message, Conversation } from './session-parser.js';
import type { AllFragments, DecisionFragment, InsightFragment, NarrativeFragment, VocabularyFragment } from './schemas.js';
import { emptyFragments } from './schemas.js';
import * as logger from './logger.js';

// ============================================================================
// Types
// ============================================================================

export interface MemoryEntry {
  /** When this entry was created/updated */
  timestamp: Date;
  /** Which sessions contributed to this entry */
  sessionIds: string[];
  /** The evolved fragments */
  fragments: AllFragments;
}

export interface MemoryIndex {
  /** Version for future migrations */
  version: number;
  /** When the index was last updated */
  lastUpdated: Date;
  /** Per-project memory entries */
  projects: Record<string, ProjectMemory>;
  /** Global vocabulary across all projects */
  globalVocabulary: VocabularyFragment | null;
  /** Cursor: timestamp of oldest unprocessed session */
  processingCursor?: Date;
  /** Global configuration settings */
  config?: GlobalConfig;
}

export interface GlobalConfig {
  /** Should verbatim artifacts be generated? */
  artifacts?: boolean;
  /** Where should artifacts be stored? 'vault' or 'local' */
  mirrorMode?: 'vault' | 'local';
  /** File extensions to index for source code search */
  sourceExtensions?: string[];
  /** Automatically re-index source during 'evolve' when git HEAD changes */
  autoIndexSource?: boolean;
  /** Cosine similarity threshold for vector search (0.0-1.0) */
  vectorThreshold?: number;
  /** API key for Jina embeddings (global fallback) */
  jinaApiKey?: string;
  /** API key for OpenRouter LLM (global fallback) */
  openRouterApiKey?: string;
  /** Generic LLM API key (global fallback) */
  llmApiKey?: string;
  /** OpenAI-compatible LLM base URL (e.g. http://127.0.0.1:1234/v1) */
  llmBaseUrl?: string;
  /** LLM model id (e.g. qwen3.8-9b-heretic-uncensored) */
  llmModel?: string;
}

export interface SessionProcessingState {
  sessionId: string;
  /** Number of messages processed */
  messageCount: number;
  /** File modification time when last processed */
  modifiedTime: Date;
  /** File size in bytes when last processed (for fast append-only check) */
  fileSize?: number;
}

export interface ProjectMemory {
  /** Project path/name */
  project: string;
  /** Current evolved fragments for this project (result of horizontal evolution) */
  current: AllFragments;
  /** When last updated */
  lastUpdated: Date;
  /** Sessions that have been processed (with state for incremental updates) */
  processedSessions: SessionProcessingState[];
  /** Absolute path to project root */
  rootPath?: string;
  /** Per-session fragment snapshots (for horizontal evolution) */
  sessionSnapshots: Array<{
    sessionId: string;
    timestamp: Date;
    fragments: AllFragments;
  }>;
  /** Historical snapshots (optional, for time travel) */
  history?: MemorySnapshot[];
  /** Projects linked for cross-project context injection */
  linkedProjects?: string[];
}

export interface MemorySnapshot {
  timestamp: Date;
  fragments: AllFragments;
  sessionId: string;
}

// ============================================================================
// Storage Paths
// ============================================================================

/**
 * Get the default memory storage directory
 */
export function getMemoryDir(): string {
  const oldDir = join(homedir(), '.claude-prose');
  const newDir = join(homedir(), '.prose');

  if (!existsSync(newDir) && existsSync(oldDir)) {
    try {
      console.log(`🚚 Migrating memory from ${oldDir} to ${newDir}...`);
      renameSync(oldDir, newDir);
    } catch (error: any) {
      console.error(`⚠️  Failed to migrate memory directory: ${error.message}`);
      return oldDir; // Fallback to old dir if migration fails
    }
  }

  return newDir;
}

/**
 * Get the path to the memory index file
 */
export function getIndexPath(): string {
  return join(getMemoryDir(), 'memory-index.json');
}

/**
 * Check if the memory vault is a Git repository
 */
export function isVaultRepo(): boolean {
  const memoryDir = getMemoryDir();
  return existsSync(join(memoryDir, '.git'));
}

/**
 * Commit changes to the memory vault if it's a Git repo
 */
export function commitToVault(message: string): void {
  if (!isVaultRepo()) return;

  try {
    const memoryDir = getMemoryDir();
    // Use git add . to capture all changes (index + projects)
    execSync(`git -C "${memoryDir}" add .`, { stdio: 'ignore' });

    // Check if there are changes to commit
    const status = execSync(`git -C "${memoryDir}" status --porcelain`, { encoding: 'utf-8' });
    if (status.trim()) {
      execSync(`git -C "${memoryDir}" commit -m "${message.replace(/"/g, '\\"')}"`, { stdio: 'ignore' });
    }
  } catch (error: any) {
    // Don't crash on git errors, just log them
    console.error(`⚠️  Vault commit failed: ${error.message}`);
  }
}

/**
 * Sanitize a file path into a project name string.
 * Replaces both / and \ with - and removes leading dash.
 */
export function sanitizePath(path: string): string {
  return path.replace(/[/\\]/g, '-').replace(/^-+/, '');
}

export function getProjectMemoryPath(projectName: string): string {
  const sanitized = projectName.replace(/[^a-zA-Z0-9-_]/g, '_');
  return join(getMemoryDir(), 'projects', `${sanitized}.json`);
}

/**
 * Get the path to a project's vector file
 */
export function getProjectVectorPath(projectName: string): string {
  const sanitized = projectName.replace(/[^a-zA-Z0-9-_]/g, '_');
  return join(getMemoryDir(), 'projects', `${sanitized}.vectors.json`);
}

/**
 * Generate a stable hash for a fragment
 */
export function calculateFragmentHash(type: string, content: string, context?: string): string {
  const data = `${type}:${content}:${context || ''}`;
  return createHash('sha256').update(data).digest('hex');
}

export function getSourceManifestPath(projectName: string): string {
  const sanitized = projectName.replace(/[^a-zA-Z0-9-_]/g, '_');
  return join(getMemoryDir(), 'projects', `${sanitized}.source.json`);
}

export function getSourceVectorPath(projectName: string): string {
  const sanitized = projectName.replace(/[^a-zA-Z0-9-_]/g, '_');
  return join(getMemoryDir(), 'projects', `${sanitized}.source-vectors.json`);
}

// ============================================================================
// Chronicle — a live, append-only dev-feed of freeform beats.
//
// Deliberately NOT a fragment: chronicle entries live outside AllFragments and
// are NEVER touched by evolution. Evolution reconciles/compresses/ages-out
// fragments; a chronicle is the opposite — discrete, timestamped, append-only,
// never overwritten. The content is a vibe, not a schema; the envelope below is
// the only structure, and exists purely so the log can be stored and rendered
// as a feed.
// ============================================================================

export interface ChronicleEntry {
  ts: string;       // ISO timestamp
  emoji?: string;
  title: string;
  body?: string;
}

export function getChroniclePath(projectName: string): string {
  const sanitized = projectName.replace(/[^a-zA-Z0-9-_]/g, '_');
  return join(getMemoryDir(), 'projects', `${sanitized}.chronicle.json`);
}

export function loadChronicle(projectName: string): ChronicleEntry[] {
  const path = getChroniclePath(projectName);
  if (!existsSync(path)) return [];
  const raw = readFileSync(path, 'utf-8');
  return JSON.parse(raw) as ChronicleEntry[];
}

export function appendChronicleEntry(projectName: string, entry: ChronicleEntry): void {
  const path = getChroniclePath(projectName);
  const projectsDir = dirname(path);
  if (!existsSync(projectsDir)) {
    mkdirSync(projectsDir, { recursive: true });
  }
  const entries = loadChronicle(projectName);
  entries.push(entry);
  writeFileSync(path, JSON.stringify(entries, null, 2));
}

// ============================================================================
// Loading & Saving
// ============================================================================

/**
 * Load the memory index
 */
export function loadMemoryIndex(): MemoryIndex {
  const indexPath = getIndexPath();

  if (!existsSync(indexPath)) {
    return {
      version: 1,
      lastUpdated: new Date(),
      projects: {},
      globalVocabulary: null,
    };
  }

  try {
    const content = readFileSync(indexPath, 'utf-8');
    const parsed = JSON.parse(content);

    // Convert date strings back to Date objects
    const index: MemoryIndex = {
      ...parsed,
      lastUpdated: new Date(parsed.lastUpdated),
      processingCursor: parsed.processingCursor ? new Date(parsed.processingCursor) : undefined,
    };

    // Also convert project-level dates
    for (const project of Object.values(index.projects)) {
      if (project.lastUpdated) {
        project.lastUpdated = new Date(project.lastUpdated);
      }
    }

    return index;
  } catch (error) {
    console.error('Failed to load memory index:', error);
    return {
      version: 1,
      lastUpdated: new Date(),
      projects: {},
      globalVocabulary: null,
    };
  }
}

/**
 * Get effective global configuration (Env > Saved Config > Defaults)
 */
/** Default file extensions for source code indexing */
export const DEFAULT_SOURCE_EXTENSIONS = ['.ts', '.js', '.py', '.go'];

/** Default cosine similarity threshold for vector search */
export const DEFAULT_VECTOR_THRESHOLD = 0.6;

export function getGlobalConfig(): GlobalConfig {
  const index = loadMemoryIndex();
  const saved = index.config || {};

  return {
    artifacts: process.env.PROSE_ARTIFACTS !== undefined
      ? process.env.PROSE_ARTIFACTS === 'true'
      : (saved.artifacts !== undefined ? saved.artifacts : true),
    mirrorMode: (process.env.PROSE_MIRROR_MODE as any) || saved.mirrorMode || 'vault',
    sourceExtensions: saved.sourceExtensions || DEFAULT_SOURCE_EXTENSIONS,
    autoIndexSource: saved.autoIndexSource !== undefined ? saved.autoIndexSource : true,
    vectorThreshold: saved.vectorThreshold !== undefined ? saved.vectorThreshold : DEFAULT_VECTOR_THRESHOLD,
    jinaApiKey: saved.jinaApiKey,
    openRouterApiKey: saved.openRouterApiKey,
    llmApiKey: saved.llmApiKey,
    llmBaseUrl: saved.llmBaseUrl,
    llmModel: saved.llmModel,
  };
}

/**
 * Save global configuration settings
 */
export function saveGlobalConfig(config: GlobalConfig): void {
  const index = loadMemoryIndex();
  index.config = { ...index.config, ...config };
  saveMemoryIndex(index);
}

/**
 * Get API key with priority: environment variable > global config
 * @param keyType - 'jina' | 'openrouter' | 'llm'
 * @returns API key or undefined if not found
 */
export function getApiKey(keyType: 'jina' | 'openrouter' | 'llm'): string | undefined {
  const config = getGlobalConfig();

  if (keyType === 'jina') {
    return process.env.PROSE_JINA_API_KEY || config.jinaApiKey;
  }

  if (keyType === 'openrouter') {
    return process.env.OPENROUTER_API_KEY || config.openRouterApiKey;
  }

  // LLM fallback chain: env vars first, then global config
  return process.env.PROSE_API_KEY ||
         process.env.LLM_API_KEY ||
         process.env.OPENROUTER_API_KEY ||
         process.env.OPENAI_API_KEY ||
         config.llmApiKey ||
         config.openRouterApiKey;
}

export function loadSourceManifest(projectName: string): any | null {
  const path = getSourceManifestPath(projectName);
  if (!existsSync(path)) return null;
  try {
    return JSON.parse(readFileSync(path, 'utf-8'));
  } catch {
    return null;
  }
}

export function saveSourceManifest(manifest: any): void {
  const path = getSourceManifestPath(manifest.project);
  writeFileSync(path, JSON.stringify(manifest, null, 2));
}

export function loadSourceVectors(projectName: string): number[][] {
  const path = getSourceVectorPath(projectName);
  if (!existsSync(path)) return [];
  try {
    return JSON.parse(readFileSync(path, 'utf-8'));
  } catch {
    return [];
  }
}

export function saveSourceVectors(projectName: string, vectors: number[][]): void {
  const path = getSourceVectorPath(projectName);
  writeFileSync(path, JSON.stringify(vectors));
}

/**
 * Save the memory index
 */
export function saveMemoryIndex(index: MemoryIndex): void {
  const indexPath = getIndexPath();
  const dir = dirname(indexPath);

  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  index.lastUpdated = new Date();
  writeFileSync(indexPath, JSON.stringify(index, null, 2));

  // Auto-commit to vault
  commitToVault('Update memory index');
}

/**
 * Load a project's memory
 */
export function loadProjectMemory(projectName: string): ProjectMemory | null {
  const memoryPath = getProjectMemoryPath(projectName);

  if (!existsSync(memoryPath)) {
    return null;
  }

  try {
    const content = readFileSync(memoryPath, 'utf-8');
    const parsed = JSON.parse(content);

    // Convert date strings back to Date objects
    // Note: sourceLinks removed - it was write-only provenance data causing vault bloat
    const { sourceLinks: _ignored, ...rest } = parsed;
    return {
      ...rest,
      lastUpdated: new Date(parsed.lastUpdated),
      processedSessions: (parsed.processedSessions || []).map((s: any) => ({
        ...s,
        modifiedTime: new Date(s.modifiedTime),
      })),
      sessionSnapshots: (parsed.sessionSnapshots || []).map((s: any) => ({
        ...s,
        timestamp: new Date(s.timestamp),
      })),
      history: (parsed.history || []).map((h: any) => ({
        ...h,
        timestamp: new Date(h.timestamp),
      })),
    };
  } catch (error) {
    console.error(`Failed to load project memory for ${projectName}:`, error);
    return null;
  }
}

/**
 * Save a project's memory
 */
export function saveProjectMemory(memory: ProjectMemory): void {
  const memoryPath = getProjectMemoryPath(memory.project);
  const dir = dirname(memoryPath);

  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  memory.lastUpdated = new Date();
  writeFileSync(memoryPath, JSON.stringify(memory, null, 2));

  // Auto-commit to vault
  commitToVault(`Evolve: ${memory.project}`);
}

/**
 * Load a project's vectors
 */
export function loadProjectVectors(projectName: string): Record<string, number[]> {
  const vectorPath = getProjectVectorPath(projectName);

  if (!existsSync(vectorPath)) {
    return {};
  }

  try {
    const content = readFileSync(vectorPath, 'utf-8');
    return JSON.parse(content);
  } catch (error) {
    console.error(`Failed to load vectors for ${projectName}:`, error);
    return {};
  }
}

/**
 * Save a project's vectors
 */
export function saveProjectVectors(projectName: string, vectors: Record<string, number[]>): void {
  const vectorPath = getProjectVectorPath(projectName);
  const dir = dirname(vectorPath);

  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  writeFileSync(vectorPath, JSON.stringify(vectors, null, 2));
}

// ============================================================================
// Memory Operations
// ============================================================================

/**
 * Create a new project memory
 */
export function createProjectMemory(project: string): ProjectMemory {
  return {
    project,
    current: emptyFragments(),
    lastUpdated: new Date(),
    processedSessions: [],
    sessionSnapshots: [],
    linkedProjects: [],
    history: [],
  };
}

/**
 * Update project memory with new evolved fragments from a session
 * This stores the session's fragments as a snapshot for horizontal evolution
 */
export function updateProjectMemory(
  memory: ProjectMemory,
  fragments: AllFragments,
  sessionId: string,
  messageCount: number,
  modifiedTime: Date,
  fileSize?: number,
  rootPath?: string
): ProjectMemory {
  // Update or add session processing state
  const existingIndex = memory.processedSessions.findIndex(s => typeof s === 'object' && s !== null && 'sessionId' in s && (s as any).sessionId === sessionId);
  const newState: SessionProcessingState = {
    sessionId,
    messageCount,
    modifiedTime,
    fileSize,
  };

  const processedSessions = existingIndex >= 0
    ? [
      ...memory.processedSessions.slice(0, existingIndex),
      newState,
      ...memory.processedSessions.slice(existingIndex + 1),
    ]
    : [...memory.processedSessions, newState];

  // Update or add session snapshot
  const snapshotIndex = memory.sessionSnapshots?.findIndex(s => s.sessionId === sessionId) ?? -1;
  const newSnapshot = {
    sessionId,
    timestamp: modifiedTime,
    fragments,
  };

  const sessionSnapshots = snapshotIndex >= 0
    ? [
      ...(memory.sessionSnapshots || []).slice(0, snapshotIndex),
      newSnapshot,
      ...(memory.sessionSnapshots || []).slice(snapshotIndex + 1),
    ]
    : [...(memory.sessionSnapshots || []), newSnapshot];

  return {
    ...memory,
    // Don't update current directly - that's done by horizontal evolution
    lastUpdated: new Date(),
    processedSessions,
    sessionSnapshots,
    rootPath: rootPath || memory.rootPath,
  };
}

/**
 * Update the current (horizontally evolved) fragments
 */
export function updateCurrentFragments(
  memory: ProjectMemory,
  current: AllFragments
): ProjectMemory {
  return {
    ...memory,
    current,
    lastUpdated: new Date(),
  };
}

/**
 * Check if a session needs processing (new or has new messages)
 */
export function getSessionProcessingState(
  memory: ProjectMemory | null,
  sessionId: string
): SessionProcessingState | null {
  if (!memory) return null;
  return (memory.processedSessions.find(s => typeof s === 'object' && s !== null && 'sessionId' in s && (s as any).sessionId === sessionId) as SessionProcessingState) || null;
}

/**
 * Check if a session needs (re)processing
 * Returns true if: never processed, or has new messages since last processing
 */
/**
 * Fast check using file size (no parsing needed for append-only JSONL)
 */
export function sessionNeedsProcessingFast(
  memory: ProjectMemory | null,
  sessionId: string,
  currentFileSize: number
): boolean {
  const state = getSessionProcessingState(memory, sessionId);
  if (!state) return true; // Never processed
  if (!state.fileSize) return true; // No file size stored, need to process

  // Append-only: if file grew, there's new content
  return currentFileSize > state.fileSize;
}

/**
 * Full check (requires parsing - use sessionNeedsProcessingFast first)
 */
export function sessionNeedsProcessing(
  memory: ProjectMemory | null,
  sessionId: string,
  currentMessageCount: number,
  currentModifiedTime: Date
): boolean {
  const state = getSessionProcessingState(memory, sessionId);
  if (!state) return true; // Never processed

  // Check if file has grown (new messages)
  if (currentMessageCount > state.messageCount) return true;

  // Check if file was modified (fallback) - convert to timestamps for reliable comparison
  const currentTime = new Date(currentModifiedTime).getTime();
  const storedTime = new Date(state.modifiedTime).getTime();
  if (currentTime > storedTime) return true;

  return false;
}

/**
 * Legacy check - kept for backwards compatibility
 */
export function isSessionProcessed(memory: ProjectMemory | null, sessionId: string): boolean {
  if (!memory) return false;
  return memory.processedSessions.some(s => s.sessionId === sessionId);
}

// ============================================================================
// Search / Query
// ============================================================================

export interface SearchResult {
  project: string;
  type: 'decision' | 'insight' | 'gotcha' | 'narrative' | 'quote' | 'source';
  content: string;
  context?: string;
  score: number;
  timestamp?: Date;
  sourceLink?: SourceLink;
  filePath?: string; // For source chunks
}

/**
 * Search across all project memories
 * Searches sessionSnapshots for temporal awareness - newer results score higher
 */
import { getJinaEmbeddings, cosineSimilarity } from './jina.js';

export async function searchMemory(query: string, options?: {
  projects?: string[];
  types?: SearchResult['type'][];
  limit?: number;
  jinaApiKey?: string;
  all?: boolean;
}): Promise<SearchResult[]> {
  const config = getGlobalConfig();
  const index = loadMemoryIndex();
  const results: SearchResult[] = [];
  const seen = new Set<string>(); // Dedupe by content hash

  // Use configured threshold (default 0.6) converted to 0-100 scale
  const vectorThreshold = (config.vectorThreshold ?? DEFAULT_VECTOR_THRESHOLD) * 100;

  const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);
  const projectFilter = options?.projects;
  const typeFilter = options?.types;

  // Hybrid search: short queries (1-2 words) use keyword matching, longer queries use semantic
  const wordCount = query.trim().split(/\s+/).length;
  const useKeywordSearch = wordCount <= 2;

  // Perform semantic search only for longer queries (3+ words) and if API key is available
  let queryVector: number[] | null = null;
  if (!useKeywordSearch && options?.jinaApiKey) {
    try {
      const embeddings = await getJinaEmbeddings([query], options.jinaApiKey, { task: 'retrieval.query' });
      if (embeddings.length > 0) {
        queryVector = embeddings[0];
      }
    } catch (e: any) {
      logger.warn(`Semantic search failed, falling back to keywords: ${e.message}`);
    }
  }

  // Find the time range for recency scoring
  let oldestTime = Date.now();
  let newestTime = 0;

  for (const projectName of Object.keys(index.projects)) {
    const memory = loadProjectMemory(projectName);
    if (!memory?.sessionSnapshots) continue;
    for (const snapshot of memory.sessionSnapshots) {
      const time = new Date(snapshot.timestamp).getTime();
      if (time < oldestTime) oldestTime = time;
      if (time > newestTime) newestTime = time;
    }
  }

  const timeRange = newestTime - oldestTime || 1; // Avoid division by zero

  for (const [projectName, projectInfo] of Object.entries(index.projects)) {
    // CWD and Linked-Project Awareness
    const isCurrentOrLinked = projectFilter && projectFilter.includes(projectName);
    if (!options?.all && projectFilter && !isCurrentOrLinked) {
      continue;
    }

    const memory = loadProjectMemory(projectName);
    const hasSnapshots = (memory?.sessionSnapshots?.length ?? 0) > 0;
    const vectors = hasSnapshots ? loadProjectVectors(projectName) : {};

    // Search through sessionSnapshots for temporal awareness
    for (const snapshot of memory?.sessionSnapshots || []) {
      const timestamp = new Date(snapshot.timestamp);
      const recencyBonus = ((timestamp.getTime() - oldestTime) / timeRange) * 20; // 0-20 bonus for recency

      const searchFragments = (type: SearchResult['type'], items: any[], getContent: (item: any) => string, getContext?: (item: any) => string) => {
        for (const item of items) {
          const content = getContent(item);
          const context = getContext ? getContext(item) : undefined;
          const hash = calculateFragmentHash(type, content, context);

          if (seen.has(hash)) continue;

          let score = 0;
          let hasVector = false;

          // Vector Score (primary - use exclusively if available)
          if (queryVector && vectors[hash]) {
            const similarity = cosineSimilarity(queryVector, vectors[hash]);
            score = similarity * 100;
            hasVector = true;
          }

          // Keyword Score (fallback only when no vector)
          if (!hasVector) {
            const keywordScore = scoreMatch(queryTerms, `${content} ${context || ''}`);
            score = keywordScore * 10;
          }

          // Threshold: use configured vectorThreshold for vectors, or ALL query terms must match for keywords
          // scoreMatch gives 15 per term, we multiply by 10, so threshold = terms * 150
          const minKeywordScore = queryTerms.length * 150;
          const threshold = hasVector ? vectorThreshold : minKeywordScore;
          if (score >= threshold) {
            seen.add(hash);
            results.push({
              project: projectName,
              type,
              content,
              context,
              score: score + recencyBonus,
              timestamp,
            });
          }
        }
      };

      // Search decisions
      if (!typeFilter || typeFilter.includes('decision')) {
        searchFragments('decision', snapshot.fragments.decisions?.decisions || [], d => d.what, d => d.why);
      }

      // Search insights
      if (!typeFilter || typeFilter.includes('insight')) {
        searchFragments('insight', snapshot.fragments.insights?.insights || [], i => i.learning, i => i.context);
        searchFragments('gotcha', snapshot.fragments.insights?.gotchas || [], g => g.issue, g => g.solution);
      }

      // Search narrative
      if (!typeFilter || typeFilter.includes('narrative')) {
        searchFragments('narrative', snapshot.fragments.narrative?.story_beats || [], b => b.summary, b => b.beat_type);
      }

      // Search quotes
      if (!typeFilter || typeFilter.includes('quote')) {
        searchFragments('quote', snapshot.fragments.narrative?.memorable_quotes || [], q => q.quote, q => q.speaker);
      }
    }

    // --- Search current (evolved/merged) fragments ---
    // These represent the "ground truth" after horizontal evolution
    if (memory?.current) {
      const currentTimestamp = new Date(memory.lastUpdated || newestTime);
      const currentRecencyBonus = 20; // Max recency bonus for evolved content

      const searchCurrentFragments = (type: SearchResult['type'], items: any[], getContent: (item: any) => string, getContext?: (item: any) => string) => {
        for (const item of items) {
          const content = getContent(item);
          const context = getContext ? getContext(item) : undefined;
          const hash = calculateFragmentHash(type, content, context);

          if (seen.has(hash)) continue;

          let score = 0;
          let hasVector = false;

          // Vector Score (primary - use exclusively if available)
          if (queryVector && vectors[hash]) {
            const similarity = cosineSimilarity(queryVector, vectors[hash]);
            score = similarity * 100;
            hasVector = true;
          }

          // Keyword Score (fallback only when no vector)
          if (!hasVector) {
            const keywordScore = scoreMatch(queryTerms, `${content} ${context || ''}`);
            score = keywordScore * 10;
          }

          const minKeywordScore = queryTerms.length * 150;
          const threshold = hasVector ? vectorThreshold : minKeywordScore;
          if (score >= threshold) {
            seen.add(hash);
            results.push({
              project: projectName,
              type,
              content,
              context,
              score: score + currentRecencyBonus,
              timestamp: currentTimestamp,
            });
          }
        }
      };

      if (!typeFilter || typeFilter.includes('decision')) {
        searchCurrentFragments('decision', memory.current.decisions?.decisions || [], d => d.what, d => d.why);
      }
      if (!typeFilter || typeFilter.includes('insight')) {
        searchCurrentFragments('insight', memory.current.insights?.insights || [], i => i.learning, i => i.context);
        searchCurrentFragments('gotcha', memory.current.insights?.gotchas || [], g => g.issue, g => g.solution);
      }
      if (!typeFilter || typeFilter.includes('narrative')) {
        searchCurrentFragments('narrative', memory.current.narrative?.story_beats || [], b => b.summary, b => b.beat_type);
      }
      if (!typeFilter || typeFilter.includes('quote')) {
        searchCurrentFragments('quote', memory.current.narrative?.memorable_quotes || [], q => q.quote, q => q.speaker);
      }
    }

    // --- Search source code chunks ---
    if (!typeFilter || typeFilter.includes('source')) {
      const sourceManifest = loadSourceManifest(projectName);
      const sourceVectorData = loadSourceVectors(projectName) as any;

      if (sourceManifest && sourceVectorData && sourceVectorData.vectors) {
        // Build hash-to-index map for source vectors
        const hashToVectorIndex = new Map<string, number>();
        for (let i = 0; i < sourceVectorData.hashes.length; i++) {
          hashToVectorIndex.set(sourceVectorData.hashes[i], i);
        }

        for (const [relativePath, fileMeta] of Object.entries(sourceManifest.files as Record<string, any>)) {
          for (const chunk of fileMeta.chunks) {
            const hash = chunk.hash;
            if (seen.has(hash)) continue;

            let score = 0;
            let hasVector = false;

            // Vector Score (primary - use exclusively if available)
            const vectorIndex = hashToVectorIndex.get(hash);
            if (queryVector && vectorIndex !== undefined) {
              const similarity = cosineSimilarity(queryVector, sourceVectorData.vectors[vectorIndex]);
              score = similarity * 100;
              hasVector = true;
            }

            // Keyword Score (fallback only when no vector)
            if (!hasVector) {
              const keywordScore = scoreMatch(queryTerms, `${relativePath} ${chunk.type} ${chunk.content || ''}`);
              score = keywordScore * 10;
            }

            // Threshold: use configured vectorThreshold for vectors, or ALL query terms must match for keywords
            const minKeywordScore = queryTerms.length * 150;
            const threshold = hasVector ? vectorThreshold : minKeywordScore;
            if (score >= threshold) {
              seen.add(hash);
              results.push({
                project: projectName,
                type: 'source',
                content: `Chunk in ${relativePath} (${chunk.type}, L${chunk.startLine}-${chunk.endLine})`,
                context: relativePath,
                score: score, // No recency bonus for source for now, or maybe use lastIndexed?
                filePath: relativePath,
              });
            }
          }
        }
      }
    }
  }

  // Sort by score descending
  results.sort((a, b) => b.score - a.score);

  // Apply limit
  if (options?.limit) {
    return results.slice(0, options.limit);
  }

  return results;
}

/**
 * Score a match against query terms
 */
function scoreMatch(queryTerms: string[], text: string): number {
  const textLower = text.toLowerCase();
  let score = 0;

  for (const term of queryTerms) {
    if (textLower.includes(term)) {
      score += 10;
      // Bonus for exact word match
      if (new RegExp(`\\b${term}\\b`).test(textLower)) {
        score += 5;
      }
    }
  }

  return score;
}

// ============================================================================
// Markdown Generation (for slash command injection)
// ============================================================================

/**
 * Generate a Markdown context file from project memory
 * This can be symlinked as a Claude Code slash command
 */
export function generateContextMarkdown(projectName: string): string | null {
  const memory = loadProjectMemory(projectName);
  if (!memory) return null;

  const lines: string[] = [
    `# Project Memory: ${projectName.replace(/^-Users-[^-]+-src-/, '')}`,
    '',
    `> Auto-generated semantic memory from ${memory.processedSessions.length} sessions`,
    `> Last updated: ${memory.lastUpdated.toLocaleString()}`,
    '',
  ];

  // Current Focus
  if (memory.current.focus?.current_goal) {
    lines.push('## Current Focus');
    lines.push('');
    lines.push(`**Goal:** ${memory.current.focus.current_goal}`);
    if (memory.current.focus.active_tasks?.length) {
      lines.push('');
      lines.push('**Active tasks:**');
      for (const task of memory.current.focus.active_tasks) {
        lines.push(`- ${task}`);
      }
    }
    if (memory.current.focus.blockers?.length) {
      lines.push('');
      lines.push('**Blockers:**');
      for (const blocker of memory.current.focus.blockers) {
        lines.push(`- ⚠️ ${blocker}`);
      }
    }
    lines.push('');
  }

  // Decisions
  if (memory.current.decisions?.decisions?.length) {
    lines.push('## Key Decisions');
    lines.push('');
    for (const decision of memory.current.decisions.decisions) {
      lines.push(`### ${decision.what}`);
      lines.push('');
      lines.push(`**Why:** ${decision.why}`);
      if (decision.alternatives?.length) {
        lines.push('');
        lines.push(`**Alternatives considered:** ${decision.alternatives.join(', ')}`);
      }
      lines.push(`**Confidence:** ${decision.confidence}`);
      lines.push('');
    }
  }

  // Insights
  if (memory.current.insights?.insights?.length) {
    lines.push('## Insights');
    lines.push('');
    for (const insight of memory.current.insights.insights) {
      lines.push(`- **${insight.learning}**`);
      if (insight.context) {
        lines.push(`  - Context: ${insight.context}`);
      }
    }
    lines.push('');
  }

  // Gotchas
  if (memory.current.insights?.gotchas?.length) {
    lines.push('## Gotchas & Pitfalls');
    lines.push('');
    for (const gotcha of memory.current.insights.gotchas) {
      lines.push(`- ⚠️ **${gotcha.issue}**`);
      if (gotcha.solution) {
        lines.push(`  - Solution: ${gotcha.solution}`);
      }
    }
    lines.push('');
  }

  // Technologies & Files (from vocabulary)
  if (memory.current.vocabulary) {
    if (memory.current.vocabulary.technologies?.length) {
      lines.push('## Technologies');
      lines.push('');
      lines.push(memory.current.vocabulary.technologies.join(', '));
      lines.push('');
    }
  }

  return lines.join('\n');
}

/**
 * Redact sensitive information like API keys and tokens from a string
 */
export function redactSecrets(text: string): string {
  if (!text) return text;

  // Patterns for common secrets
  const patterns = [
    // OpenAI, Anthropic, OpenRouter, etc.
    /(sk-[a-zA-Z0-9]{20,})/g,
    /(g-sk-[a-zA-Z0-9]{20,})/g,
    /(x-pk-[a-zA-Z0-9]{20,})/g,
    // Generic tokens/keys in assignments/headers
    /(?:key|token|password|secret|auth|api-key|apikey)["']?\s*[:=]\s*["']?([a-zA-Z0-9\-_]{16,})["']?/ig,
    // Authorization headers
    /(?:Authorization:\s*(?:Bearer|Basic)\s+)([a-zA-Z0-9\.\-_]{16,})/ig
  ];

  let redacted = text;
  for (const pattern of patterns) {
    redacted = redacted.replace(pattern, (match, secret) => {
      // If there's a capturing group (secret), replace only that
      if (secret) {
        return match.replace(secret, '[REDACTED]');
      }
      // Otherwise replace the whole match (for simple patterns)
      return '[REDACTED]';
    });
  }

  return redacted;
}

/**
 * Generate a verbatim Markdown record of a session
 */
export function generateVerbatimMarkdown(conversation: Conversation): string {
  const date = new Date(conversation.startTime || new Date());
  const dateStr = date.toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  });
  const timeStr = date.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
  });

  const lines: string[] = [
    `# Session: ${conversation.sessionId.slice(0, 8)}`,
    '',
    `**Date:** ${dateStr} at ${timeStr}`,
    `**Project:** ${conversation.project}`,
    `**Session ID:** \`${conversation.sessionId}\``,
    '',
    '---',
    '',
  ];

  for (const msg of conversation.messages) {
    const roleName = msg.role === 'user' ? 'Designer' : 'Claude';
    lines.push(`**${roleName}:**`);
    lines.push(msg.content);
    lines.push('');
  }

  lines.push('---');
  lines.push('');
  lines.push(`*Generated by [prose](https://github.com/6digit/prose) - Digital Archaeology Mode*`);

  return lines.join('\n');
}

/**
 * Write a verbatim session artifact to the Vault (default) or specified directory
 */
export function writeVerbatimSessionArtifact(conversation: Conversation, outputDir?: string): string {
  // Default to vault mirrors: ~/.prose/mirrors/[project]/
  let dir = outputDir;
  if (!dir) {
    const vaultDir = getMemoryDir();
    const projectName = conversation.project;
    dir = join(vaultDir, 'mirrors', projectName);
  }

  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  const filename = `session-${conversation.sessionId.slice(0, 8)}.md`;
  const outputPath = join(dir, filename);

  const rawMarkdown = generateVerbatimMarkdown(conversation);
  const redactedMarkdown = redactSecrets(rawMarkdown);

  writeFileSync(outputPath, redactedMarkdown);

  return outputPath;
}

/**
 * Generate a Markdown summary for a single session snapshot
 */
export function generateSessionMarkdown(snapshot: { sessionId: string; timestamp: Date | string; fragments: AllFragments }, projectName: string): string {
  const date = new Date(snapshot.timestamp);
  const dateStr = date.toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  });
  const timeStr = date.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
  });

  const lines: string[] = [
    `# Session Snapshot: ${snapshot.sessionId.slice(0, 8)}`,
    '',
    `**Project:** ${projectName.replace(/^-Users-[^-]+-src-/, '')}`,
    `**Date:** ${dateStr} at ${timeStr}`,
    `**Session ID:** \`${snapshot.sessionId}\``,
    '',
  ];

  // Focus
  const focus = snapshot.fragments.focus;
  if (focus?.current_goal) {
    const focusEmoji = focus.current_goal.toLowerCase().includes('fix') ? '🔧' :
      focus.current_goal.toLowerCase().includes('feat') ? '✨' : '🎯';
    lines.push(`## ${focusEmoji} Focus`);
    lines.push('');
    lines.push(`**Goal:** ${focus.current_goal}`);
    if (focus.active_tasks?.length) {
      lines.push('');
      lines.push('**Active tasks:**');
      for (const task of focus.active_tasks) {
        lines.push(`- ${task}`);
      }
    }
    if (focus.blockers?.length) {
      lines.push('');
      lines.push('**Blockers:**');
      for (const blocker of focus.blockers) {
        lines.push(`- ⚠️ ${blocker}`);
      }
    }
    lines.push('');
  }

  // Story Beats
  const beats = snapshot.fragments.narrative?.story_beats || [];
  if (beats.length) {
    lines.push('## 📖 The Story');
    lines.push('');
    for (const beat of beats) {
      const typeStr = beat.beat_type ? `[${beat.beat_type.toUpperCase()}] ` : '';
      lines.push(`- ${typeStr}${beat.summary}${beat.emotional_tone ? ` *(${beat.emotional_tone})*` : ''}`);
    }
    lines.push('');
  }

  // Quotes
  const quotes = snapshot.fragments.narrative?.memorable_quotes || [];
  if (quotes.length) {
    lines.push('## 💬 Memorable Quotes');
    lines.push('');
    for (const q of quotes) {
      lines.push(`> "${q.quote}"`);
      lines.push(`> — *${q.speaker}*`);
      lines.push('');
    }
  }

  // Decisions
  const decisions = snapshot.fragments.decisions?.decisions || [];
  if (decisions.length) {
    lines.push('## ⚖️ Decisions Made');
    lines.push('');
    for (const d of decisions) {
      lines.push(`### ${d.what}`);
      lines.push('');
      lines.push(`**Why:** ${d.why}`);
      lines.push(`**Confidence:** ${d.confidence}`);
      lines.push('');
    }
  }

  // Insights
  const insights = snapshot.fragments.insights?.insights || [];
  const gotchas = snapshot.fragments.insights?.gotchas || [];
  if (insights.length || gotchas.length) {
    lines.push('## 💡 Insights & Gotchas');
    lines.push('');
    for (const i of insights) {
      lines.push(`- **Learning:** ${i.learning}`);
      if (i.context) lines.push(`  - *Context:* ${i.context}`);
    }
    for (const g of gotchas) {
      lines.push(`- ⚠️ **Issue:** ${g.issue}`);
      if (g.solution) lines.push(`  - *Solution:* ${g.solution}`);
    }
    lines.push('');
  }

  return lines.join('\n');
}

/**
 * Export a session snapshot as a Markdown artifact (Summarized version)
 * Note: We are moving towards verbatim mirroring as the primary artifact.
 */
export function exportSessionSnapshotArtifact(snapshot: { sessionId: string; timestamp: Date | string; fragments: AllFragments }, projectName: string, outputDir: string): string {
  if (!existsSync(outputDir)) {
    mkdirSync(outputDir, { recursive: true });
  }

  const filename = `summary-${snapshot.sessionId.slice(0, 8)}.md`;
  const outputPath = join(outputDir, filename);
  const markdown = generateSessionMarkdown(snapshot, projectName);
  writeFileSync(outputPath, markdown);
  return outputPath;
}

/**
 * Write context markdown to a file (for slash command symlink)
 */
export function writeContextFile(projectName: string, outputPath: string): boolean {
  const markdown = generateContextMarkdown(projectName);
  if (!markdown) return false;

  const dir = dirname(outputPath);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  writeFileSync(outputPath, markdown);
  return true;
}

/**
 * Get memory statistics
 */
export function getMemoryStats(): {
  totalProjects: number;
  totalDecisions: number;
  totalInsights: number;
  totalSessions: number;
  lastUpdated: Date | null;
} {
  const index = loadMemoryIndex();

  let totalDecisions = 0;
  let totalInsights = 0;
  let totalSessions = 0;

  for (const projectName of Object.keys(index.projects)) {
    const memory = loadProjectMemory(projectName);
    if (memory) {
      totalDecisions += memory.current.decisions?.decisions?.length || 0;
      totalInsights += memory.current.insights?.insights?.length || 0;
      totalSessions += memory.processedSessions.length;
    }
  }

  return {
    totalProjects: Object.keys(index.projects).length,
    totalDecisions,
    totalInsights,
    totalSessions,
    lastUpdated: index.lastUpdated,
  };
}
