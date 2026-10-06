#!/usr/bin/env node
/**
 * Prose CLI - Semantic memory for AI development
 *
 * Commands:
 *   evolve   - Process sessions and evolve fragments
 *   search   - Semantic search through memory
 *   grep     - Verbatim regex search across parsed agent sessions
 *   status   - Show memory statistics
 *   show     - Display current fragments for a project
 */

// Load .env file if present (check multiple locations)
import { config } from 'dotenv';
import { homedir } from 'os';
import { join, basename, dirname } from 'path';
import { existsSync, mkdirSync, writeFileSync, readdirSync, appendFileSync, symlinkSync, readFileSync } from 'fs';
import { execSync } from 'child_process';
config({ quiet: true });  // Load from current directory
config({ path: join(homedir(), '.config', 'prose', '.env'), quiet: true });  // Global config

import { Command } from 'commander';
import { discoverSessionFiles, parseSessionFile, parseSessionFileFromOffset, getSessionStats, getClaudeProjectsDir, type Message, type SessionFile } from './session-parser.js';
import { discoverCodexSessionFiles, parseCodexSessionFile, parseCodexSessionFileFromOffset } from './codex-session-parser.js';
import { discoverDevinSessionFiles, parseDevinSessionFile } from './devin-session-parser.js';
import { evolveAllFragments } from './evolve.js';
import { emptyFragments, type AllFragments } from './schemas.js';
import {
  loadMemoryIndex,
  saveMemoryIndex,
  loadProjectMemory,
  saveProjectMemory,
  createProjectMemory,
  updateProjectMemory,
  updateCurrentFragments,
  sessionNeedsProcessing,
  sessionNeedsProcessingFast,
  getSessionProcessingState,
  searchMemory,
  getMemoryStats,
  generateContextMarkdown,
  writeContextFile,
  writeVerbatimSessionArtifact,
  sanitizePath,
  getMemoryDir,
  isVaultRepo,
  commitToVault,
  loadProjectVectors,
  saveProjectVectors,
  calculateFragmentHash,
  getGlobalConfig,
  saveGlobalConfig,
  loadSourceManifest,
  getApiKey,
  getChroniclePath,
  loadChronicle,
  appendChronicleEntry,
  type ChronicleEntry,
} from './memory.js';
import { getLlmConfig, hasLlmAccess } from './llm.js';
import {
  loadChronicleConfig,
  getChronicleConfigPath,
  buildContent,
  emitToDiscord,
} from './sinks.js';
import { getJinaEmbeddings, cosineSimilarity } from './jina.js';
import { evolveHorizontal } from './horizontal.js';
import { generateWebsite } from './web.js';
import { indexProjectSource } from './source-indexer.js';
import { getGitCommits, isGitRepo, getAntigravityBrains, getAntigravityArtifacts, parseAntigravityArtifact, matchBrainToProject, getLatestGitCommitDate, getDesignSessions, parseDesignSession } from './source-parsers.js';
import { startServer } from './server.js';
import { injectMemory, ensureTemplate, initSkillFile, writeSkillFile } from './injector.js';
import { startDesignSession } from './design.js';
import { addFragment, detectProject, type FragmentType } from './add.js';
import { snap } from './snap.js';
import { whisper } from './whisper.js';
import { gossip } from './gossip.js';
import { standup, parseDuration } from './standup.js';
import { grep, escapeRegex } from './grep.js';
import { stats, renderCsv } from './stats.js';
import { session, SessionAmbiguousError, SessionNotFoundError } from './session.js';
import { tail, NoSessionForCwdError } from './tail.js';
import { setBaton, listBatons, clearBatons, renderBatonLine, batonOriginNote } from './baton.js';
import type { SourceType } from './session-parser.js';
import * as logger from './logger.js';
import { createRequire } from 'module';

// Single source of truth for the version — the hardcoded string drifted
// (`prose --version` reported 0.4.0 while the package shipped 0.8.0).
const pkg = createRequire(import.meta.url)('../package.json') as { version: string };

const program = new Command();

program
  .name('prose')
  .description('Semantic memory for AI development - extract, evolve, and query the meaning of your collaboration')
  .version(pkg.version)
  .option('--api-key <key>', 'Override LLM API key')
  .option('-v, --verbose', 'Show detailed progress')
  .option('-q, --quiet', 'Suppress unnecessary output')
  .option('--trace', 'Show extremely detailed debugging logs')
  .on('option:verbose', () => logger.setLogLevel(logger.LogLevel.VERBOSE))
  .on('option:quiet', () => logger.setLogLevel(logger.LogLevel.ERROR))
  .on('option:trace', () => logger.setLogLevel(logger.LogLevel.TRACE));

/**
 * Helper to detect the current project based on CWD
 */
function detectProjectFromCwd(): string | undefined {
  const cwd = process.cwd();
  const cwdSanitized = sanitizePath(cwd);

  // 1. Check evolved memory index
  const index = loadMemoryIndex();
  const indexMatch = Object.keys(index.projects).find(p =>
    p === cwdSanitized || p.endsWith(cwdSanitized) || cwdSanitized.endsWith(p.replace(/^-/, ''))
  );
  if (indexMatch) return indexMatch;

  // 2. Check Claude's projects directory for recent sessions
  const projectsDir = getClaudeProjectsDir();
  if (existsSync(projectsDir)) {
    const projectDirs = readdirSync(projectsDir, { withFileTypes: true })
      .filter(d => d.isDirectory())
      .map(d => d.name);

    const match = projectDirs.find(p => {
      const dirName = p.replace(/^-/, '');
      return dirName === cwdSanitized ||
        dirName.endsWith(cwdSanitized) ||
        cwdSanitized.endsWith(dirName);
    });
    if (match) return match;
  }

  // 3. Check Codex sessions for this CWD
  const codexSessions = discoverCodexSessionFiles(cwd);
  if (codexSessions.length > 0) {
    return codexSessions[0].project;
  }

  // 4. Fallback: check session files discovered across all projects
  const sessions = discoverSessionFiles();
  const projectNames = [...new Set(sessions.map(s => s.project))];
  return projectNames.find(p =>
    p === cwdSanitized || p.endsWith(cwdSanitized) || cwdSanitized.endsWith(p.replace(/^-/, ''))
  );
}

/**
 * Format a project name for display
 */
function formatProjectName(name: string | undefined): string {
  if (!name) return 'Unknown';
  return name.replace(/^-Users-[^-]+-src-/, '').replace(/^-/, '');
}

// ============================================================================
// init - Set up prose in a project
// ============================================================================

program
  .command('init')
  .description('Initialize prose in the current project')
  .argument('[subcommand]', 'Subcommand: "hooks" to install PreCompact hook')
  .action((subcommand) => {
    const cwd = process.cwd();
    const projectName = basename(cwd);

    if (subcommand === 'hooks') {
      // Install PreCompact hook
      console.log(`🔧 Installing PreCompact hook for: ${projectName}\n`);

      const claudeDir = '.claude';
      const settingsPath = `${claudeDir}/settings.local.json`;

      // Check if settings.local.json already exists
      if (existsSync(settingsPath)) {
        console.log(`⚠️  ${settingsPath} already exists.`);
        console.log('');
        console.log('To add the hook manually, add this to your settings.local.json:');
        console.log('');
        console.log(`  "hooks": {
    "PreCompact": [
      {
        "matcher": "manual",
        "hooks": [
          {
            "type": "command",
            "command": "prose evolve &",
            "timeout": 10
          }
        ]
      }
    ]
  }`);
        return;
      }

      // Create .claude directory if needed
      if (!existsSync(claudeDir)) {
        mkdirSync(claudeDir, { recursive: true });
      }

      // Create settings.local.json with PreCompact hook
      const settings = {
        hooks: {
          PreCompact: [
            {
              matcher: 'manual',
              hooks: [
                {
                  type: 'command',
                  command: 'prose evolve &',
                  timeout: 10,
                },
              ],
            },
          ],
        },
      };

      writeFileSync(settingsPath, JSON.stringify(settings, null, 2));

      logger.info(`✅ Created ${settingsPath}`);
      logger.info('');
      logger.info('Now when you run /compact in Claude Code, prose will');
      logger.info('automatically evolve your session memory in the background.');
      console.log('');
      console.log('💡 Note: settings.local.json is gitignored - each developer opts in individually.');
      return;
    }

    // Default init behavior
    logger.info(`🧠 Initializing prose for: ${projectName}\n`);

    // 1. Create the flash skill (idempotent)
    initSkillFile(cwd, projectName);

    // 2. .gitignore protection for local prose data
    const gitignorePath = join(cwd, '.gitignore');
    const proseIgnore = '.claude/prose/';
    const skillIgnore = '.claude/skills/flash/';

    if (existsSync(gitignorePath)) {
      let gitignoreContent = readFileSync(gitignorePath, 'utf-8');
      let modified = false;

      if (!gitignoreContent.includes(proseIgnore)) {
        gitignoreContent += `\n# Prose local data\n${proseIgnore}\n`;
        modified = true;
      }
      if (!gitignoreContent.includes(skillIgnore)) {
        gitignoreContent += `# Prose skill (auto-generated)\n${skillIgnore}\n`;
        modified = true;
      }

      if (modified) {
        writeFileSync(gitignorePath, gitignoreContent);
        logger.info(`🛡️  Updated .gitignore with prose entries`);
      } else {
        logger.info(`ℹ️  .gitignore already has prose entries`);
      }
    } else if (isGitRepo(cwd)) {
      writeFileSync(gitignorePath, `# Prose local data\n${proseIgnore}\n# Prose skill (auto-generated)\n${skillIgnore}\n`);
      logger.info(`🛡️  Created .gitignore with prose entries`);
    }

    // 3. Create local mirror directory (optional, for session artifacts)
    const localMirrorDir = join(cwd, '.claude', 'prose');
    if (!existsSync(localMirrorDir)) {
      mkdirSync(localMirrorDir, { recursive: true });
    }

    const vaultDir = getMemoryDir();
    const vaultMirrorPath = join(vaultDir, 'mirrors', projectName);
    const localSymlinkPath = join(localMirrorDir, 'mirrors');

    if (!existsSync(localSymlinkPath)) {
      try {
        if (!existsSync(vaultMirrorPath)) {
          mkdirSync(vaultMirrorPath, { recursive: true });
        }
        symlinkSync(vaultMirrorPath, localSymlinkPath, 'dir');
        logger.info(`🔗 Linked .claude/prose/mirrors -> Vault`);
      } catch (e) {
        // Symlinks may fail on some systems - not critical
      }
    }

    console.log('');
    console.log('📝 Next steps:');
    logger.info('   1. Run: prose evolve');
    logger.info('   2. Run: prose init hooks  (optional: auto-evolve on /compact)');
    logger.info('');
    logger.info('💡 The flash skill will auto-activate when Claude needs project context.');
  });

// ============================================================================
// evolve - Process sessions and evolve fragments
// ============================================================================

program
  .command('evolve')
  .description('Process Claude Code sessions and evolve semantic fragments')
  .option('-p, --project <path>', 'Filter to specific project path')
  .option('-l, --limit <n>', 'Limit number of sessions to process', '50')
  .option('-f, --force', 'Reprocess already-processed sessions')
  .option('--dry-run', 'Show what would be processed without making changes')
  .option('-v, --verbose', 'Show detailed progress')
  .option('--trace', 'Show detailed decision tracing for debugging')
  .option('--git', 'Include git commits in evolution', true)
  .option('--no-git', 'Exclude git commits')
  .option('--antigravity', 'Include Antigravity artifacts in evolution', true)
  .option('--no-antigravity', 'Exclude Antigravity artifacts')
  .option('--artifacts', 'Export per-session Markdown artifacts', true)
  .option('--no-artifacts', 'Disable per-session artifact export')
  .action(async (options) => {
    const config = getGlobalConfig();
    if (!hasLlmAccess(options.apiKey)) {
      logger.error('No LLM access. Set an API key (OPENROUTER_API_KEY / "prose config set openrouter-api-key <key>") or a local endpoint ("prose config set llm-base-url http://127.0.0.1:1234/v1")');
      process.exit(1);
    }
    const apiKey = options.apiKey || getLlmConfig().apiKey || 'prose-local';
    const jinaApiKey = getApiKey('jina');

    // Resolve artifacts preference (Option > Global Config)
    const shouldMirror = options.artifacts !== undefined ? options.artifacts : config.artifacts;

    logger.info('🧬 Prose - Evolving semantic memory');
    logger.warn('⚠️  ALPHA / EXPERIMENTAL: This tool is largely untested. Use at your own risk.');
    logger.warn('💸 COST NOTICE: Evolution performs multiple LLM passes. Monitor token usage.\n');

    // Auto-detect project from current directory if not specified
    let projectFilter = options.project;
    if (!projectFilter) {
      projectFilter = detectProjectFromCwd();
      if (projectFilter) {
        logger.info(`📁 Evolving: ${formatProjectName(projectFilter)}\n`);
      }
    }

    // Discover sessions
    const sessions = discoverSessionFiles(projectFilter, process.cwd());
    const codexSessions = discoverCodexSessionFiles(projectFilter);
    sessions.push(...codexSessions);
    // Devin sessions carry their real cwd in sessions.db — same project filter.
    sessions.push(...discoverDevinSessionFiles(projectFilter));

    // Add Git if requested
    if (options.git) {
      const cwd = process.cwd();
      if (isGitRepo(cwd)) {
        const repoName = cwd.split('/').pop() || 'repo';
        const project = projectFilter || sanitizePath(cwd);
        const latestCommitDate = getLatestGitCommitDate(cwd);
        sessions.push({
          path: cwd,
          sessionId: `git-${repoName}`,
          project,
          modifiedTime: latestCommitDate || new Date(),
          fileSize: 1000, // Placeholder
          sourceType: 'git',
        });
      }
    }

    // Add Antigravity if requested
    if (options.antigravity && projectFilter) {
      const brains = getAntigravityBrains();
      const matchingBrains = brains.filter(b => matchBrainToProject(b, projectFilter!));

      for (const brain of matchingBrains) {
        const artifacts = getAntigravityArtifacts(brain, projectFilter!);
        for (const art of artifacts) {
          // Tag as antigravity sourceType
          sessions.push({ ...art, sourceType: 'antigravity' });
        }
      }

      // Add Intelligent Design sessions
      if (projectFilter) {
        const designSessions = getDesignSessions(process.cwd(), projectFilter);
        for (const ds of designSessions) {
          sessions.push({ ...ds, sourceType: 'design' as any });
        }
      }
    }

    const limit = parseInt(options.limit, 10);

    // Load memory index
    const index = loadMemoryIndex();

    // First pass: FAST check using file size (no parsing!)
    // Separate into: needs actual work vs just needs fileSize backfill
    const sessionsNeedingWork: typeof sessions = [];
    const trace = options.trace;

    for (const session of sessions) {
      // Skip zero-size files
      if (session.fileSize === 0) {
        logger.trace(`${session.sessionId.slice(0, 8)}: skip (zero-size file)`);
        continue;
      }

      // Load project memory for fast check
      const memory = loadProjectMemory(session.project);
      const state = getSessionProcessingState(memory, session.sessionId);
      if (!state) {
        logger.trace(`${session.sessionId.slice(0, 8)}: NEW (no prior state)`);
        sessionsNeedingWork.push(session);
      } else if (state.fileSize === undefined) {
        logger.trace(`${session.sessionId.slice(0, 8)}: baseline (has messageCount=${state.messageCount}, no fileSize)`);
        sessionsNeedingWork.push(session);
      } else if (options.force || session.fileSize > state.fileSize) {
        logger.trace(`${session.sessionId.slice(0, 8)}: ${options.force ? 'FORCE' : 'UPDATED'} (fileSize ${state.fileSize} -> ${session.fileSize})`);
        sessionsNeedingWork.push(session);
      } else {
        if (trace) logger.trace(`  [TRACE] ${session.sessionId.slice(0, 8)}: skip (up to date)`);
      }
    }

    // Sort actual work oldest-first for temporal evolution, then apply limit
    const sessionsOldestFirst = [...sessionsNeedingWork].sort((a, b) =>
      a.modifiedTime.getTime() - b.modifiedTime.getTime()
    );
    const sessionsToProcess = sessionsOldestFirst.slice(0, limit);

    if (options.verbose) {
      logger.info(`📁 Found ${sessions.length} sessions, ${sessionsNeedingWork.length} need work, processing ${sessionsToProcess.length}`);
    }

    let processed = 0;
    let totalTokens = 0;

    for (const session of sessionsToProcess) {
      const projectName = session.project;

      // Load or create project memory
      let memory = loadProjectMemory(projectName) || createProjectMemory(projectName);
      const prevState = getSessionProcessingState(memory, session.sessionId);

      // Use offset-based parsing if we have a stored fileSize (append-only optimization)
      let messagesToProcess: Message[] = [];
      let totalMessageCount: number;
      let lastProcessedBytes: number = prevState?.fileSize || 0;
      let isIncremental = false;

      if (trace) {
        logger.trace(`=== Processing ${session.sessionId.slice(0, 8)} ===`);
        logger.trace(`prevState: ${prevState ? `messageCount=${prevState.messageCount}, fileSize=${prevState.fileSize ?? 'undefined'}` : 'null'}`);
        logger.trace(`session: fileSize=${session.fileSize}`);
      }

      if (session.sourceType === 'git') {
        logger.info(`📖 Syncing git log... (${session.sessionId})`);
        messagesToProcess = getGitCommits(session.path, 10);
        totalMessageCount = messagesToProcess.length; // Assuming getGitCommits returns all relevant commits
      } else if (session.sourceType === ('design' as any)) {
        const designMessages = parseDesignSession(session.path, session.sessionId);
        totalMessageCount = designMessages.length;
        messagesToProcess = prevState ? designMessages.slice(prevState.messageCount) : designMessages;
        if (messagesToProcess.length > 0) {
          logger.info(`📖 Syncing design session... (${session.sessionId})`);
        }
      } else if (session.sourceType === 'antigravity') {
        const artifactMessages = parseAntigravityArtifact(session.path, session.sessionId, projectName);
        totalMessageCount = artifactMessages.length;
        messagesToProcess = prevState ? artifactMessages.slice(prevState.messageCount) : artifactMessages;
        if (messagesToProcess.length > 0) {
          console.log(`📖 Ingesting Antigravity artifact: ${basename(session.path)}...`);
        }
      } else if (session.sourceType === 'devin') {
        // The shared sessions.db grows as a whole; byte-offset reads don't
        // apply, so always full-parse and slice on message count.
        const devinMessages = parseDevinSessionFile(session.path, session.sessionId).messages;
        totalMessageCount = devinMessages.length;
        messagesToProcess = prevState ? devinMessages.slice(prevState.messageCount) : devinMessages;
        if (messagesToProcess.length > 0) {
          console.log(`📖 Ingesting Devin session: ${session.sessionId}...`);
        }
      } else {
        const isCodex = session.sourceType === 'codex';
        const prevFileSize = prevState?.fileSize;
        const canIncremental = typeof prevFileSize === 'number' &&
          prevFileSize < session.fileSize &&
          (!isCodex || session.path.endsWith('.jsonl'));

        if (canIncremental && prevState) {
          // FAST PATH: Only read new bytes from the file
          if (trace) console.log(`  [TRACE] -> FAST PATH: byte-offset read from ${prevFileSize}`);
          const result = isCodex
            ? parseCodexSessionFileFromOffset(
                session.path,
                prevFileSize,
                session.sessionId,
                projectName
              )
            : parseSessionFileFromOffset(
                session.path,
                prevFileSize,
                session.sessionId,
                projectName
              );
          messagesToProcess = result.messages;
          totalMessageCount = prevState.messageCount + result.messages.length;
          lastProcessedBytes = result.processedBytes;
          isIncremental = true;
          console.log(`📖 Updating ${session.sessionId.slice(0, 8)}... (+${result.messages.length} new messages, read ${result.processedBytes - prevFileSize} bytes)`);
        } else {
          // FULL PARSE: New session or no stored fileSize
          if (trace) console.log(`  [TRACE] -> FULL PARSE: ${prevState?.fileSize ? 'fileSize unchanged or smaller' : 'no stored fileSize'}`);
          const conversation = isCodex ? parseCodexSessionFile(session.path) : parseSessionFile(session.path);
          totalMessageCount = conversation.messages.length;
          lastProcessedBytes = conversation.processedBytes;
          if (trace) console.log(`  [TRACE] parsed ${totalMessageCount} messages from file, processedBytes=${lastProcessedBytes}`);

          if (prevState && prevState.messageCount < conversation.messages.length) {
            // Had previous state but no fileSize - slice from message count
            if (trace) console.log(`  [TRACE] -> INCREMENTAL: prevState.messageCount(${prevState.messageCount}) < parsed(${totalMessageCount})`);
            messagesToProcess = conversation.messages.slice(prevState.messageCount);
            isIncremental = true;
            console.log(`📖 Updating ${session.sessionId.slice(0, 8)}... (+${messagesToProcess.length} new messages)`);
          } else if (prevState && prevState.messageCount >= conversation.messages.length) {
            // No new messages - but backfill fileSize for fast check optimization
            if (trace) console.log(`  [TRACE] -> SKIP: prevState.messageCount(${prevState.messageCount}) >= parsed(${totalMessageCount})`);
            if (!prevState.fileSize || !memory.rootPath) {
              if (trace) console.log(`  [TRACE] -> backfilling fileSize=${session.fileSize}, rootPath=${process.cwd()}`);
              prevState.fileSize = session.fileSize;
              memory.rootPath = process.cwd();
              saveProjectMemory(memory);
            }
            console.log(`📖 Processing ${session.sessionId.slice(0, 8)}... (no new messages, skipping)`);
            continue;
          } else {
            // New session
            if (trace) console.log(`  [TRACE] -> NEW SESSION: no prevState`);
            messagesToProcess = conversation.messages;
            console.log(`📖 Processing ${session.sessionId.slice(0, 8)}... (${projectName.slice(-30)})`);
          }
        }
      }

      // Write verbatim artifacts FIRST (for all sessions, regardless of new messages)
      // This ensures digital archaeology captures every conversation
      if (shouldMirror && !['git', 'antigravity', 'design', 'devin', 'perplexity'].includes(session.sourceType as string)) {
        try {
          const fullConversation = session.sourceType === 'codex'
            ? parseCodexSessionFile(session.path)
            : parseSessionFile(session.path);

          // Security check: If writing to repo, ensure it's ignored
          if (config.mirrorMode === 'local' && isGitRepo(process.cwd())) {
            try {
              execSync('git check-ignore -q .claude/prose/', { stdio: 'ignore' });
            } catch (e) {
              console.log('⚠️  SECURITY WARNING: .claude/prose/ is not gitignored. Sessions may be committed accidentally.');
              console.log('   Run: prose init  (to fix .gitignore automatically)');
            }
          }

          const outputDir = config.mirrorMode === 'local' ? join(process.cwd(), '.claude', 'prose') : undefined;
          writeVerbatimSessionArtifact(fullConversation, outputDir);
        } catch (e: any) {
          console.log(`   ⚠️  Failed to write artifact: ${e.message}`);
        }
      }

      if (messagesToProcess.length === 0) {
        if (trace) console.log('  [TRACE] -> NO MESSAGES: updating metadata to skip next time');
        // Update metadata anyway so we don't keep picking this session up as "new/unprocessed"
        memory = updateProjectMemory(
          memory,
          emptyFragments(), // Start with empty if it's new and empty
          session.sessionId,
          totalMessageCount,
          session.modifiedTime,
          lastProcessedBytes
        );
        saveProjectMemory(memory);
        continue;
      }

      if (options.dryRun) {
        console.log('   [dry-run] Would evolve fragments');
        continue;
      }

      // Window size for evolution - Gemini Flash has a huge context, we can process 2000 messages at once
      const windowSize = 2000;

      const windows = [];
      for (let i = 0; i < messagesToProcess.length; i += windowSize) {
        windows.push(messagesToProcess.slice(i, i + windowSize));
      }

      // Rolling window: feed old fragments + new messages → evolved fragments
      // For incremental: load previous session snapshot
      // For new session: start empty
      const existingSnapshot = memory.sessionSnapshots?.find(s => s.sessionId === session.sessionId);
      let currentFragments = existingSnapshot?.fragments || emptyFragments();

      for (let i = 0; i < windows.length; i++) {
        const window = windows[i];

        if (options.verbose) {
          console.log(`   🔄 Window ${i + 1}/${windows.length} (${window.length} messages)`);
        }

        const result = await evolveAllFragments(
          currentFragments,
          {
            messages: window,
            allFragments: currentFragments,
            project: projectName,
            sessionId: session.sessionId,
          },
          { apiKey, jinaApiKey }
        );

        if (result.errors.length > 0 && options.verbose) {
          for (const error of result.errors) {
            console.log(`   ⚠️  ${error}`);
          }
        }

        currentFragments = result.fragments;
        totalTokens += result.tokensUsed;
      }

      // Update memory with message count and file size for incremental tracking
      memory = updateProjectMemory(
        memory,
        currentFragments,
        session.sessionId,
        totalMessageCount,
        session.modifiedTime,
        lastProcessedBytes,
        process.cwd()
      );
      saveProjectMemory(memory);

      // Update index
      index.projects[projectName] = {
        lastUpdated: new Date().toISOString() as unknown as Date,
      } as any;

      processed++;
      console.log(`   ✅ Evolved (${windows.length} windows, ${totalTokens} tokens total)`);

    }

    // Save index
    if (!options.dryRun) {
      saveMemoryIndex(index);
    }

    console.log('\n📊 Summary:');
    console.log(`   Found: ${sessions.length} sessions, ${sessionsNeedingWork.length} need updates`);
    console.log(`   Processed: ${processed} sessions`);
    if (sessionsNeedingWork.length > sessionsToProcess.length) {
      console.log(`   ⏳ Remaining: ${sessionsNeedingWork.length - sessionsToProcess.length} sessions need work (limit reached)`);
    }
    console.log(`   Tokens used: ${totalTokens}`);

    // Run horizontal evolution if we processed any sessions OR if the target project is stale relative to its links
    const targetProjectName = projectFilter;
    const targetMemory = targetProjectName ? loadProjectMemory(targetProjectName) : null;
    let isStale = false;
    if (targetMemory && (targetMemory.linkedProjects || []).length > 0) {
      for (const link of targetMemory.linkedProjects!) {
        const linkedMemory = loadProjectMemory(link);
        if (linkedMemory && linkedMemory.lastUpdated > targetMemory.lastUpdated) {
          isStale = true;
          break;
        }
      }
    }

    if ((processed > 0 || isStale) && !options.dryRun) {
      if (isStale && processed === 0) {
        logger.info(`🔄 Link Update: Project ${formatProjectName(targetProjectName)} is stale relative to its links. Re-evolving...`);
      } else {
        console.log('\n🔄 Running horizontal evolution...');
      }

      // Get all unique projects that were processed, plus the target if it's stale
      const projectsProcessed = [...new Set(sessionsToProcess.map(s => s.project))];
      if (targetProjectName && isStale && !projectsProcessed.includes(targetProjectName)) {
        projectsProcessed.push(targetProjectName);
      }

      for (const projectName of projectsProcessed) {
        const memory = loadProjectMemory(projectName);
        if (!memory || !memory.sessionSnapshots?.length) continue;

        // "Old data is the OPPOSITE of evolution"
        // Feed only what we JUST processed into the horizontal evolution step.
        const currentProject = projectName;
        const newSessionIds = new Set(sessionsToProcess
          .filter(s => s.project === currentProject)
          .map(s => s.sessionId));

        const newSnapshots = memory.sessionSnapshots.filter(s => newSessionIds.has(s.sessionId));

        let snapshotsToUse = newSnapshots;
        if (snapshotsToUse.length === 0) {
          if (projectName === targetProjectName && isStale) {
            // Use the last 3 snapshots as context for re-evolution with new links
            snapshotsToUse = memory.sessionSnapshots
              .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
              .slice(0, 3);

            if (options.verbose) {
              logger.info(`   🔄 Using ${snapshotsToUse.length} existing snapshots for re-contextualization`);
            }
          }
        }

        if (snapshotsToUse.length === 0) {
          if (options.verbose) {
            console.log(`   ⏭️ Skipping horizontal evolution for ${projectName}: No snapshots found`);
          }
          continue;
        }

        // Load linked projects
        const linkedFragments: Array<{ projectName: string; fragments: AllFragments }> = [];
        for (const linkedName of (memory.linkedProjects || [])) {
          const linkedMemory = loadProjectMemory(linkedName);
          if (linkedMemory) {
            linkedFragments.push({
              projectName: linkedName,
              fragments: linkedMemory.current
            });
          }
        }

        const result = await evolveHorizontal(
          snapshotsToUse,
          {
            apiKey,
            windowSize: Math.max(snapshotsToUse.length, 3),
            currentFragments: memory.current,
            externalFragments: linkedFragments
          }
        );

        const updated = updateCurrentFragments(memory, result.current);
        saveProjectMemory(updated);

        // Update skill file and CLAUDE.md (if template exists) for cwd project
        const cwdSanitized = sanitizePath(process.cwd());
        if (projectName === cwdSanitized || projectName.endsWith(`-${cwdSanitized}`)) {
          writeSkillFile(process.cwd(), updated, projectName);
          injectMemory(process.cwd(), updated);
        }

        console.log(`   ${formatProjectName(projectName)}: ${result.sessionsIncluded} sessions → current`);
        if (result.musings) {
          console.log(`   💭 ${result.musings.slice(0, 100)}...`);
        }
        totalTokens += result.tokensUsed;
      }
    }

    console.log(`   Total tokens: ${totalTokens}`);
    console.log(`   Memory stored: ${getMemoryDir()}`);

    // Auto-backfill vectors for architectural memory (decisions, insights, etc.)
    if (!options.dryRun && jinaApiKey && processed > 0) {
      const detectedProject = projectFilter || detectProjectFromCwd();
      if (detectedProject) {
        const memory = loadProjectMemory(detectedProject);
        if (memory) {
          const vectors = loadProjectVectors(detectedProject);
          const toEmbed: { hash: string; text: string }[] = [];

          // Helper to collect fragments
          const collectFragments = (fragments: any) => {
            for (const d of fragments.decisions?.decisions || []) {
              const hash = calculateFragmentHash('decision', d.what, d.why);
              if (!vectors[hash]) toEmbed.push({ hash, text: `${d.what} ${d.why}` });
            }
            for (const i of fragments.insights?.insights || []) {
              const hash = calculateFragmentHash('insight', i.learning, i.context);
              if (!vectors[hash]) toEmbed.push({ hash, text: `${i.learning} ${i.context}` });
            }
            for (const g of fragments.insights?.gotchas || []) {
              const hash = calculateFragmentHash('gotcha', g.issue, g.solution);
              if (!vectors[hash]) toEmbed.push({ hash, text: `${g.issue} ${g.solution}` });
            }
            for (const b of fragments.narrative?.story_beats || []) {
              const hash = calculateFragmentHash('narrative', b.summary, b.beat_type);
              if (!vectors[hash]) toEmbed.push({ hash, text: b.summary });
            }
            for (const q of fragments.narrative?.memorable_quotes || []) {
              const hash = calculateFragmentHash('quote', q.quote, q.speaker);
              if (!vectors[hash]) toEmbed.push({ hash, text: `${q.quote} - ${q.speaker}` });
            }
          };

          // Collect from session snapshots and current
          for (const snapshot of memory.sessionSnapshots || []) {
            collectFragments(snapshot.fragments);
          }
          if (memory.current) {
            collectFragments(memory.current);
          }

          if (toEmbed.length > 0) {
            console.log(`\n🧠 Backfilling ${toEmbed.length} memory vectors...`);
            try {
              const batchSize = 50;
              for (let i = 0; i < toEmbed.length; i += batchSize) {
                const batch = toEmbed.slice(i, i + batchSize);
                const embeddings = await getJinaEmbeddings(batch.map(b => b.text), jinaApiKey);
                batch.forEach((item, idx) => { vectors[item.hash] = embeddings[idx]; });
              }
              saveProjectVectors(detectedProject, vectors);
              console.log(`   ✅ Memory vectors up to date`);
            } catch (e: any) {
              console.log(`   ⚠️  Memory backfill failed: ${e.message}`);
            }
          }
        }
      }
    }

    // Auto-index source code if git HEAD has changed (respects config.autoIndexSource)
    if (!options.dryRun && jinaApiKey && config.autoIndexSource !== false) {
      const cwd = process.cwd();
      const detectedProject = projectFilter || detectProjectFromCwd();

      if (detectedProject && isGitRepo(cwd)) {
        const { getGitHead, indexProjectSource } = await import('./source-indexer.js');
        const currentHead = getGitHead(cwd);
        const manifest = loadSourceManifest(detectedProject);

        if (!manifest || manifest.gitHead !== currentHead) {
          // Use configured extensions for the diff check
          const extGlobs = (config.sourceExtensions || ['.ts', '.js']).map(ext => `'*${ext}'`).join(' ');
          const changedFiles = manifest?.gitHead
            ? (() => {
              try {
                const diff = execSync(`git diff --name-only ${manifest.gitHead}..${currentHead} -- ${extGlobs}`, { encoding: 'utf-8', cwd });
                return diff.split('\n').filter(Boolean).length;
              } catch { return '?'; }
            })()
            : 'all';

          console.log(`\n🔍 Source index outdated (${changedFiles} files changed), re-indexing...`);
          try {
            const stats = await indexProjectSource(detectedProject, cwd, jinaApiKey);
            console.log(`   ✅ Indexed ${stats.filesIndexed} files, ${stats.chunksCreated} new chunks`);
          } catch (e: any) {
            console.log(`   ⚠️  Source indexing failed: ${e.message}`);
          }
        } else {
          if (options.verbose) {
            console.log(`\n🔍 Source index up to date (HEAD unchanged)`);
          }
        }
      }
    }

    // Always update skill file (and CLAUDE.md if template exists) for current project
    if (!options.dryRun) {
      const cwd = process.cwd();
      const cwdSanitized = sanitizePath(cwd);
      const index = loadMemoryIndex();
      const detectedProject = Object.keys(index.projects).find(p =>
        p === cwdSanitized || p.endsWith(cwdSanitized) || cwdSanitized.endsWith(p.replace(/^-/, ''))
      );
      if (detectedProject) {
        const memory = loadProjectMemory(detectedProject);
        if (memory) {
          writeSkillFile(cwd, memory, detectedProject);
          injectMemory(cwd, memory);
        }
      }
    }
  });

// ============================================================================
// merge - Integrate memory from another project
// ============================================================================

program
  .command('merge')
  .description('Integrate memory fragments from another project')
  .requiredOption('--from <project>', 'Source project to import from')
  .option('--to <project>', 'Target project (defaults to current)')
  .option('--dry-run', 'Show what would be merged without making changes')
  .action(async (options) => {
    if (!hasLlmAccess(options.apiKey)) {
      logger.error('No LLM access. Set an API key (OPENROUTER_API_KEY / "prose config set openrouter-api-key <key>") or a local endpoint ("prose config set llm-base-url http://127.0.0.1:1234/v1")');
      process.exit(1);
    }
    const apiKey = options.apiKey || getLlmConfig().apiKey || 'prose-local';

    const index = loadMemoryIndex();

    // Resolve source project
    const sourceQuery = options.from;
    const sourceProject = Object.keys(index.projects).find(p =>
      p === sourceQuery || p.endsWith(`-${sourceQuery}`) || p.includes(sourceQuery)
    );

    if (!sourceProject) {
      logger.error(`Source project not found: ${sourceQuery}`);
      process.exit(1);
    }

    // Resolve target project
    let targetProjectQuery = options.to;
    let targetProject;
    if (!targetProjectQuery) {
      const cwd = process.cwd();
      const cwdSanitized = sanitizePath(cwd);
      targetProject = Object.keys(index.projects).find(p =>
        p === cwdSanitized || p.endsWith(cwdSanitized) || cwdSanitized.endsWith(p.replace(/^-/, ''))
      );
    } else {
      targetProject = Object.keys(index.projects).find(p =>
        p === targetProjectQuery || p.endsWith(`-${targetProjectQuery}`) || p.includes(targetProjectQuery)
      );
    }

    if (!targetProject) {
      console.error(`❌ Target project not found. Set --to or run from a project directory.`);
      process.exit(1);
    }

    if (sourceProject === targetProject) {
      console.error(`❌ Cannot merge a project into itself.`);
      process.exit(1);
    }

    console.log(`🧬 Merging: ${formatProjectName(sourceProject)} → ${formatProjectName(targetProject)}\n`);

    const sourceMemory = loadProjectMemory(sourceProject);
    const targetMemory = loadProjectMemory(targetProject);

    if (!sourceMemory) {
      console.error(`❌ Could not load source memory for ${sourceProject}`);
      process.exit(1);
    }
    if (!targetMemory) {
      console.error(`❌ Could not load target memory for ${targetProject}`);
      process.exit(1);
    }

    if (options.dryRun) {
      console.log('🧪 Dry run - would merge current fragments from source into target.');
      return;
    }

    console.log('🔄 Running integration evolution...');

    // Wrap source's current fragments as a pseudo-snapshot
    const integrationSnapshot = {
      sessionId: `integration-${sourceProject.replace(/^-/, '')}-${Date.now()}`,
      timestamp: new Date(),
      fragments: sourceMemory.current
    };

    const result = await evolveHorizontal(
      [integrationSnapshot],
      {
        apiKey,
        windowSize: 1,
        currentFragments: targetMemory.current
      }
    );

    const updated = updateCurrentFragments(targetMemory, result.current);
    saveProjectMemory(updated);

    // Update index for target
    index.projects[targetProject] = {
      ...index.projects[targetProject],
      lastUpdated: new Date()
    };
    saveMemoryIndex(index);

    // Update skill file and CLAUDE.md if target matches CWD
    const cwd = process.cwd();
    const cwdSanitized = cwd.replace(/\//g, '-').replace(/^-/, '');
    if (targetProject === cwdSanitized || targetProject.endsWith(`-${cwdSanitized}`)) {
      writeSkillFile(cwd, updated, targetProject);
      injectMemory(cwd, updated);
    }

    console.log('\n✅ Integration complete.');
    if (result.musings) {
      console.log(`💭 ${result.musings}`);
    }
  });

// ============================================================================
// link - Manage persistent cross-project links
// ============================================================================

program
  .command('link [target-project]')
  .description('Manage persistent cross-project links for integrated context')
  .option('--remove', 'Remove a link')
  .option('--list', 'List all linked projects')
  .action((targetProject, options) => {
    const currentProject = detectProjectFromCwd();
    if (!currentProject) {
      logger.error('Could not detect current project from directory');
      process.exit(1);
    }

    const memory = loadProjectMemory(currentProject);
    if (!memory) {
      logger.error(`No memory found for ${currentProject}. Run evolve first.`);
      process.exit(1);
    }

    if (options.list || !targetProject) {
      const links = memory.linkedProjects || [];
      if (links.length === 0) {
        logger.info(`Project ${formatProjectName(currentProject)} has no linked projects.`);
      } else {
        logger.info(`Linked projects for ${formatProjectName(currentProject)}:`);
        links.forEach(l => console.log(`  - ${formatProjectName(l)}`));
      }
      return;
    }

    // Direct match or partial match for target
    const index = loadMemoryIndex();
    const projectNames = Object.keys(index.projects);
    const matchedTarget = projectNames.find(p => p.toLowerCase().includes(targetProject.toLowerCase()));

    if (!matchedTarget) {
      logger.error(`Target project "${targetProject}" not found in memory index.`);
      process.exit(1);
    }

    if (options.remove) {
      const initialCount = (memory.linkedProjects || []).length;
      memory.linkedProjects = (memory.linkedProjects || []).filter(p => p !== matchedTarget);
      if ((memory.linkedProjects || []).length < initialCount) {
        saveProjectMemory(memory);
        logger.success(`Removed link: ${matchedTarget}`);
      } else {
        logger.info(`Project ${matchedTarget} was not linked.`);
      }
    } else {
      if (!memory.linkedProjects) memory.linkedProjects = [];
      if (!memory.linkedProjects.includes(matchedTarget)) {
        memory.linkedProjects.push(matchedTarget);
        saveProjectMemory(memory);
        logger.success(`Linked ${matchedTarget} -> ${currentProject}`);
      } else {
        logger.info(`Project ${matchedTarget} is already linked.`);
      }
    }
  });

// ============================================================================
// design - Start an interactive design session to shape memory
// ============================================================================

program
  .command('design')
  .description('Interactive session with the AI Architect to shape project memory')
  .option('-p, --project <path>', 'Filter to specific project path')
  .option('--model <name>', 'Model to use', 'google/gemini-3-flash-preview')
  .action(async (options) => {
    if (!hasLlmAccess(options.apiKey)) {
      logger.error('No LLM access. Set an API key (OPENROUTER_API_KEY / "prose config set openrouter-api-key <key>") or a local endpoint ("prose config set llm-base-url http://127.0.0.1:1234/v1")');
      process.exit(1);
    }
    const apiKey = options.apiKey || getLlmConfig().apiKey || 'prose-local';

    let projectFilter = options.project;
    const cwd = process.cwd();
    const cwdSanitized = sanitizePath(cwd);

    if (!projectFilter) {
      const index = loadMemoryIndex();
      projectFilter = Object.keys(index.projects).find(p =>
        p === cwdSanitized || p.endsWith(cwdSanitized) || cwdSanitized.endsWith(p.replace(/^-/, ''))
      );
    }

    if (!projectFilter) {
      logger.error('Could not detect project. Please run `prose evolve` first or specify --project');
      process.exit(1);
    }

    const memory = loadProjectMemory(projectFilter);
    if (!memory) {
      console.error(`❌ No memory found for project: ${projectFilter}`);
      process.exit(1);
    }

    await startDesignSession(projectFilter, memory, {
      apiKey,
      model: options.model,
    });
  });

// ============================================================================
// add - Direct fragment addition without evolution
// ============================================================================

program
  .command('add <type> <content>')
  .description('Add a fragment directly to project memory (decision, gotcha, insight, focus)')
  .option('-p, --project <path>', 'Project path (auto-detected from CWD)')
  .option('--why <reason>', 'Reasoning for decisions')
  .option('--solution <fix>', 'Solution for gotchas')
  .option('--context <ctx>', 'Context for insights')
  .action((type: string, content: string, options) => {
    const validTypes: FragmentType[] = ['decision', 'gotcha', 'insight', 'focus'];
    if (!validTypes.includes(type as FragmentType)) {
      logger.error(`Invalid fragment type: ${type}. Must be one of: ${validTypes.join(', ')}`);
      process.exit(1);
    }

    let projectFilter = options.project;
    if (!projectFilter) {
      projectFilter = detectProject(process.cwd());
    }

    if (!projectFilter) {
      logger.error('Could not detect project. Run `prose evolve` first or specify --project');
      process.exit(1);
    }

    try {
      addFragment(projectFilter, type as FragmentType, content, {
        why: options.why,
        solution: options.solution,
        context: options.context,
      });
    } catch (err: any) {
      logger.error(`Failed to add fragment: ${err.message}`);
      process.exit(1);
    }
  });

// ============================================================================
// search - Semantic search through memory
// ============================================================================

program
  .command('search <query>')
  .description('Semantic search through evolved memory')
  .option('-p, --project <name>', 'Filter to specific project (auto-detects from cwd if not specified)')
  .option('-a, --all', 'Search all projects (ignore cwd auto-detection)')
  .option('-t, --type <types>', 'Filter by type: decision,insight,gotcha,narrative,quote,source', 'decision,insight,gotcha,source')
  .option('-l, --limit <n>', 'Limit results', '10')
  .option('--json', 'Output as JSON')
  .action(async (query, options) => {
    const types = options.type.split(',') as any[];
    const limit = parseInt(options.limit, 10);

    // Auto-detect project from current directory unless --all is specified
    let projectFilter: string[] | undefined;
    if (options.project) {
      projectFilter = [options.project];
    } else if (!options.all) {
      // Try to match cwd to a project
      const cwd = process.cwd();
      const cwdSanitized = sanitizePath(cwd);
      const index = loadMemoryIndex();
      const matchingProject = Object.keys(index.projects).find(p =>
        p === cwdSanitized || p.endsWith(cwdSanitized) || cwdSanitized.endsWith(p.replace(/^-/, ''))
      );
      if (matchingProject) {
        projectFilter = [matchingProject];
        console.log(`📁 Searching in: ${formatProjectName(matchingProject)} (use --all for global search)\n`);
      }
    }

    const jinaApiKey = getApiKey('jina');

    const results = await searchMemory(query, {
      projects: projectFilter,
      types,
      limit,
      jinaApiKey,
      all: options.all,
    });

    if (options.json) {
      console.log(JSON.stringify(results, null, 2));
      return;
    }

    if (results.length === 0) {
      logger.info(`No results found for "${query}"`);
      return;
    }

    logger.info(`🔍 Found ${results.length} results for "${query}":\n`);

    for (const result of results) {
      const typeIcon = {
        decision: '⚖️ ',
        insight: '💡',
        gotcha: '⚠️ ',
        narrative: '📖',
        quote: '💬',
        source: '💻', // NEW: Source code icon
      }[result.type];

      const project = formatProjectName(result.project);
      const dateStr = result.timestamp
        ? new Date(result.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
        : '';
      logger.info(`${typeIcon} [${result.type}] (${project}) ${dateStr}`);
      logger.info(`   ${result.content}`);
      if (result.context) {
        console.log(`   → ${result.context}`);
      }
      console.log('');
    }
  });

// ============================================================================
// status - Show memory statistics
// ============================================================================

program
  .command('status')
  .description('Show memory statistics (CWD-aware, use "status global" for all projects)')
  .argument('[scope]', '"global" for all projects, otherwise auto-detects from cwd')
  .action((scope) => {
    const isGlobal = scope === 'global';

    // Try to detect project from CWD
    let detectedProject: string | undefined;
    if (!isGlobal) {
      detectedProject = detectProjectFromCwd();
    }

    if (detectedProject && !isGlobal) {
      // Project-specific status
      const shortName = formatProjectName(detectedProject);
      logger.info(`📊 ${shortName}\n`);

      // Raw session files for this project
      const sessions = discoverSessionFiles(detectedProject);
      const codexSessions = discoverCodexSessionFiles(detectedProject);
      const allSessions = [...sessions, ...codexSessions];
      let totalMessages = 0;
      let earliestDate: Date | null = null;
      let latestDate: Date | null = null;

      for (const session of allSessions) {
        const conversation = session.sourceType === 'codex'
          ? parseCodexSessionFile(session.path)
          : parseSessionFile(session.path);
        totalMessages += conversation.messages.length;

        if (!earliestDate || session.modifiedTime < earliestDate) {
          earliestDate = session.modifiedTime;
        }
        if (!latestDate || session.modifiedTime > latestDate) {
          latestDate = session.modifiedTime;
        }
      }

      console.log('📁 Raw Sessions:');
      console.log(`   Files: ${allSessions.length}`);
      console.log(`   Messages: ${totalMessages.toLocaleString()}`);
      if (earliestDate && latestDate) {
        const fmt = (d: Date) => d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        console.log(`   Date range: ${fmt(earliestDate)} - ${fmt(latestDate)}`);
      }

      // Evolved memory for this project
      const memory = loadProjectMemory(detectedProject);
      if (memory) {
        const decisions = memory.current.decisions?.decisions?.length || 0;
        const insights = memory.current.insights?.insights?.length || 0;
        const gotchas = memory.current.insights?.gotchas?.length || 0;
        const processedCount = memory.processedSessions?.length || 0;

        console.log('\n🧠 Evolved Memory:');
        console.log(`   Sessions: ${processedCount}/${allSessions.length} processed`);
        console.log(`   Decisions: ${decisions}`);
        console.log(`   Insights: ${insights}`);
        console.log(`   Gotchas: ${gotchas}`);
        console.log(`   Last evolved: ${memory.lastUpdated.toLocaleString()}`);
      } else {
        console.log('\n🧠 Evolved Memory:');
        logger.info('   Not yet evolved. Run: prose evolve');
      }

      logger.info(`\n💡 Tip: Use "prose status global" for all projects`);
    } else {
      // Global status with per-project breakdown
      const stats = getMemoryStats();
      const sessionStats = getSessionStats();
      const codexSessions = discoverCodexSessionFiles();
      const index = loadMemoryIndex();

      let codexMessageCount = 0;
      let codexEarliest: Date | null = null;
      let codexLatest: Date | null = null;

      for (const session of codexSessions) {
        const conversation = parseCodexSessionFile(session.path);
        codexMessageCount += conversation.messages.length;

        if (!codexEarliest || session.modifiedTime < codexEarliest) {
          codexEarliest = session.modifiedTime;
        }
        if (!codexLatest || session.modifiedTime > codexLatest) {
          codexLatest = session.modifiedTime;
        }
      }

      const rawSessions = sessionStats.totalSessions + codexSessions.length;
      const rawMessages = sessionStats.totalMessages + codexMessageCount;
      let rawEarliest = sessionStats.dateRange.earliest;
      let rawLatest = sessionStats.dateRange.latest;

      if (!rawEarliest || (codexEarliest && codexEarliest < rawEarliest)) {
        rawEarliest = codexEarliest;
      }
      if (!rawLatest || (codexLatest && codexLatest > rawLatest)) {
        rawLatest = codexLatest;
      }

      logger.info('📊 Prose - Global Status\n');

      // Per-project table
      const projects = Object.keys(index.projects).sort();
      if (projects.length > 0) {
        console.log('┌' + '─'.repeat(74) + '┐');
        console.log('│ ' + 'Project'.padEnd(28) + 'Sessions'.padStart(10) + 'Decisions'.padStart(11) + 'Insights'.padStart(10) + 'Vectors'.padStart(12) + ' │');
        console.log('├' + '─'.repeat(74) + '┤');

        let totalDecisions = 0, totalInsights = 0, totalGotchas = 0;

        for (const projectName of projects) {
          const memory = loadProjectMemory(projectName);
          if (!memory) continue;

          const shortName = formatProjectName(projectName).slice(0, 26);
          const sessions = memory.processedSessions?.length || 0;
          const decisions = memory.current.decisions?.decisions?.length || 0;
          const insights = memory.current.insights?.insights?.length || 0;
          const gotchas = memory.current.insights?.gotchas?.length || 0;

          totalDecisions += decisions;
          totalInsights += insights;
          totalGotchas += gotchas;

          // Check vector status
          const vectors = loadProjectVectors(projectName);
          const vectorCount = Object.keys(vectors).length;
          const vectorStatus = vectorCount > 0 ? `✓ ${vectorCount}` : '—';

          console.log('│ ' +
            shortName.padEnd(28) +
            sessions.toString().padStart(10) +
            decisions.toString().padStart(11) +
            (insights + gotchas).toString().padStart(10) +
            vectorStatus.padStart(12) + ' │');
        }

        console.log('└' + '─'.repeat(74) + '┘');
        console.log('');
      }

      // Summary stats
      console.log('📈 Totals:');
      console.log(`   ${stats.totalProjects} projects │ ${stats.totalSessions} sessions │ ${stats.totalDecisions} decisions │ ${stats.totalInsights} insights`);

      // Session discovery info
      if (rawEarliest && rawLatest) {
        const fmt = (d: Date) => d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        console.log(`\n📁 Raw Sessions: ${rawSessions} files (${rawMessages.toLocaleString()} messages)`);
        console.log(`   Spanning ${fmt(rawEarliest)} → ${fmt(rawLatest)}`);
      }

      if (stats.lastUpdated) {
        console.log(`\n⏱️  Last evolved: ${stats.lastUpdated.toLocaleString()}`);
      }

      console.log(`📂 Vault: ${getMemoryDir()}`);
    }
  });

// ============================================================================
// show - Display current fragments for a project
// ============================================================================

program
  .command('show [project]')
  .description('Display current fragments for a project')
  .option('--decisions', 'Show only decisions')
  .option('--insights', 'Show only insights')
  .option('--narrative', 'Show only narrative')
  .option('--json', 'Output as JSON')
  .action((project, options) => {
    // Try to detect project if not specified
    let matchedProject: string | undefined;
    const index = loadMemoryIndex();
    const projectNames = Object.keys(index.projects);

    if (project) {
      // Direct match or partial match
      matchedProject = projectNames.find(p =>
        p.toLowerCase().includes(project.toLowerCase())
      );
    } else {
      // CWD detection
      matchedProject = detectProjectFromCwd();
    }

    if (!matchedProject) {
      if (project) {
        logger.error(`Project "${project}" not found`);
      } else {
        logger.error('Could not detect project from current directory');
      }

      if (projectNames.length > 0) {
        console.log('Available projects:');
        for (const p of projectNames) {
          console.log(`  - ${formatProjectName(p)}`);
        }
      }
      process.exit(1);
    }

    const memory = loadProjectMemory(matchedProject);
    if (!memory) {
      console.error(`❌ No memory found for project "${matchedProject}"`);
      process.exit(1);
    }

    if (options.json) {
      console.log(JSON.stringify(memory.current, null, 2));
      return;
    }

    const shortName = formatProjectName(matchedProject);
    console.log(`🧠 Memory for: ${shortName}\n`);
    console.log(`   Sessions processed: ${memory.processedSessions.length}`);
    console.log(`   Last updated: ${memory.lastUpdated.toLocaleString()}\n`);

    const showAll = !options.decisions && !options.insights && !options.narrative;

    if (showAll || options.decisions) {
      console.log('⚖️  DECISIONS:');
      for (const decision of memory.current.decisions?.decisions || []) {
        console.log(`   • ${decision.what}`);
        console.log(`     Why: ${decision.why}`);
        console.log(`     Confidence: ${decision.confidence}`);
        console.log('');
      }
    }

    if (showAll || options.insights) {
      console.log('💡 INSIGHTS:');
      for (const insight of memory.current.insights?.insights || []) {
        console.log(`   • ${insight.learning}`);
        if (insight.context) {
          console.log(`     Context: ${insight.context}`);
        }
        console.log('');
      }

      if (memory.current.insights?.gotchas?.length) {
        console.log('⚠️  GOTCHAS:');
        for (const gotcha of memory.current.insights.gotchas) {
          console.log(`   • ${gotcha.issue}`);
          if (gotcha.solution) {
            console.log(`     Solution: ${gotcha.solution}`);
          }
          console.log('');
        }
      }
    }

    if (showAll || options.narrative) {
      console.log('📖 NARRATIVE:');
      for (const beat of memory.current.narrative?.story_beats || []) {
        const tone = beat.emotional_tone ? ` (${beat.emotional_tone})` : '';
        console.log(`   [${beat.beat_type}]${tone}: ${beat.summary}`);
      }
      console.log('');

      if (memory.current.narrative?.memorable_quotes?.length) {
        console.log('💬 QUOTES:');
        for (const quote of memory.current.narrative.memorable_quotes) {
          console.log(`   "${quote.quote}" - ${quote.speaker}`);
        }
      }
    }
  });

// ============================================================================
// context - Generate markdown for slash command injection
// ============================================================================

program
  .command('context [project]')
  .description('Generate context markdown for slash command injection')
  .option('-o, --output <path>', 'Output file path (default: stdout)')
  .option('--install', 'Install as Claude Code slash command in current project')
  .action((project, options) => {
    // Try to detect project if not specified
    let matchedProject: string | undefined;
    const index = loadMemoryIndex();
    const projectNames = Object.keys(index.projects);

    if (project) {
      matchedProject = projectNames.find(p =>
        p.toLowerCase().includes(project.toLowerCase())
      );
    } else {
      matchedProject = detectProjectFromCwd();
    }

    if (!matchedProject) {
      if (project) {
        logger.error(`Project "${project}" not found`);
      } else {
        logger.error('Could not detect project from current directory');
      }

      if (projectNames.length > 0) {
        console.log('Available projects:');
        for (const p of projectNames) {
          console.log(`  - ${formatProjectName(p)}`);
        }
      }
      process.exit(1);
    }

    const markdown = generateContextMarkdown(matchedProject);
    if (!markdown) {
      console.error(`❌ No memory found for project "${matchedProject}"`);
      process.exit(1);
    }

    if (options.install) {
      // Install as slash command in .claude/commands/
      const commandsDir = '.claude/commands';
      const commandPath = `${commandsDir}/memory.md`;

      writeContextFile(matchedProject, commandPath);
      console.log(`✅ Installed as slash command: /memory`);
      console.log(`   File: ${commandPath}`);
      console.log('\n   Use in Claude Code: /memory');
    } else if (options.output) {
      writeContextFile(matchedProject, options.output);
      console.log(`✅ Context written to: ${options.output}`);
    } else {
      // Output to stdout
      console.log(markdown);
    }
  });

// ============================================================================
// web - Generate browsable HTML website
// ============================================================================

program
  .command('web')
  .description('Generate a browsable HTML website from project memory')
  .option('-o, --output <dir>', 'Output directory', './prose-web')
  .option('--open', 'Open in browser after generating')
  .action((options) => {
    logger.info('🌐 Generating Prose website...\n');

    generateWebsite(options.output);

    if (options.open) {
      const indexPath = `${options.output}/index.html`;
      import('child_process').then(({ exec }) => {
        exec(`open "${indexPath}"`);
      });
    }
  });

// ============================================================================
// serve - Interactive dashboard
// ============================================================================

program
  .command('serve')
  .description('Start interactive dashboard server')
  .option('-p, --port <port>', 'Port to run on', '3000')
  .option('--open', 'Open browser automatically')
  .action(async (options) => {
    const { startServer } = await import('./server.js');
    const port = parseInt(options.port, 10);

    startServer(port);

    if (options.open) {
      setTimeout(() => {
        import('child_process').then(({ exec }) => {
          exec(`open "http://localhost:${port}"`);
        });
      }, 500);
    }
  });

// ============================================================================
// artifacts - Export per-session artifacts
// ============================================================================

program
  .command('artifacts')
  .description('Export all session fragments as Markdown artifacts')
  .option('-p, --project <path>', 'Project name or path')
  .option('--local', 'Export to local .claude/prose directory instead of Vault')
  .action(async (options) => {
    let projectName = options.project;
    if (!projectName) {
      projectName = detectProjectFromCwd();
      if (!projectName) {
        logger.error('Could not detect project from current directory');
        process.exit(1);
      }
    }

    const config = getGlobalConfig();
    const mirrorMode = options.local ? 'local' : config.mirrorMode;

    logger.info(`📦 Exporting verbatim artifacts for ${projectName} (Mode: ${mirrorMode})...`);

    const artifactsDir = mirrorMode === 'local'
      ? join(process.cwd(), '.claude', 'prose')
      : undefined; // Defaults to vault in writeVerbatimSessionArtifact

    const sessions = discoverSessionFiles(projectName);
    let count = 0;
    for (const session of sessions) {
      const conv = parseSessionFile(session.path);
      writeVerbatimSessionArtifact(conv, artifactsDir);
      count++;
    }

    const targetPath = artifactsDir || join(getMemoryDir(), 'mirrors', projectName);
    logger.success(`Exported ${count} verbatim artifacts to ${targetPath}`);
  });

// ============================================================================
// snap - Verbatim readout of recent activity in current cwd
// ============================================================================

program
  .command('snap')
  .description('Verbatim readout of recent agent sessions in the current cwd (Claude Code CLI, ACP, Codex, opencode, Cursor, pi, and OMP). Orient without retracing.')
  .option('--bytes <n>', 'Byte budget for assembled text (default 4000)', (v) => parseInt(v, 10))
  .option('--turns <n>', 'Last N messages per session (default 4)', (v) => parseInt(v, 10))
  .option('--sessions <n>', 'Max sessions to include (default 10)', (v) => parseInt(v, 10))
  .option('--max-message-bytes <n>', 'Per-message byte cap; long messages get truncated with [N bytes elided] (default 1500, 0 disables)', (v) => parseInt(v, 10))
  .option('--include-current', 'Include the actively-written session (default skipped)')
  .option('--include-sdk-cli', 'Include sdk-cli sessions (Claude Code automation: commit-message generators, etc. — noise by default)')
  .option('--cwd <path>', 'Override current working directory')
  .option('--no-batons', 'Suppress the "you are here" baton header')
  .option('--json', 'Emit JSON with metadata instead of plain text')
  .action((options) => {
    const result = snap({
      cwd: options.cwd,
      bytes: options.bytes,
      turnsPerSession: options.turns,
      maxSessions: options.sessions,
      maxMessageBytes: options.maxMessageBytes,
      includeCurrent: options.includeCurrent === true,
      includeSdkCli: options.includeSdkCli === true,
      includeBatons: options.batons !== false,
    });

    if (options.json) {
      process.stdout.write(JSON.stringify(result, null, 2) + '\n');
      return;
    }

    if (result.sessionsIncluded === 0) {
      // Two ways to be empty, and they mean opposite things to an orienting
      // agent. Saying "nothing found" for either is the failure prose exists to
      // prevent: it sends the reader off to re-derive yesterday from `git log`.
      //
      //   1. A baton is present. That IS orientation — the previous session's
      //      sign-off, often naming the next move — and it was already loaded
      //      into `result.text`. Print it; never discard it just because no
      //      *session* survived the filters.
      //   2. The only sessions here were the reader's own live one. "Nothing
      //      happened in this repo" is false; "the only recent session is the
      //      conversation you're already in" is true and actionable.
      const emitted = result.batons.length > 0;
      if (emitted) process.stdout.write(result.text);

      const own = result.skippedLiveSessions;
      const why = own > 0
        ? `No past sessions for cwd ${result.cwd} — the only recent session here is your own (${own} skipped). You already have that context.`
        : `No recent sessions found for cwd ${result.cwd}.`;
      process.stderr.write(`${why}\n`);
      // Emitting a baton is a real answer; only a truly empty read is a failure.
      process.exit(emitted ? 0 : 1);
    }

    process.stdout.write(result.text);
    const trail = `\n# snap: ${result.sessionsIncluded} session(s), ${result.turnsIncluded} message(s), ${result.bytes} bytes${result.truncated ? ' (truncated by budget)' : ''}\n`;
    process.stderr.write(trail);
  });

// ============================================================================
// whisper - Neighborhood-aware verbatim readout (no LLM)
// ============================================================================

program
  .command('whisper')
  .description('Verbatim readout of recent agent sessions across the cwd and its conceptual sibling repos (project family). Reads Claude Code CLI, ACP, Codex, opencode, Cursor, pi, and OMP. No LLM — pure read.')
  .option('--bytes <n>', 'Byte budget for the assembled verbatim text (default 4000)', (v) => parseInt(v, 10))
  .option('--turns <n>', 'Last N messages per session (default 4)', (v) => parseInt(v, 10))
  .option('--sessions <n>', 'Max sessions to include per repo (default: 5 for self, 2 for siblings)', (v) => parseInt(v, 10))
  .option('--max-message-bytes <n>', 'Per-message byte cap; long messages get truncated with [N bytes elided] (default 1500, 0 disables)', (v) => parseInt(v, 10))
  .option('--include-current', 'Include the actively-written session (default skipped)')
  .option('--include-sdk-cli', 'Include sdk-cli sessions (Claude Code automation: commit-message generators, etc. — noise by default)')
  .option('--cwd <path>', 'Override current working directory')
  .option('--cwd-only', 'Skip neighborhood expansion — collect only this cwd (no sibling repos)')
  .option('--json', 'Emit JSON with metadata + per-member blocks instead of plain text')
  .action((options) => {
    const result = whisper({
      cwd: options.cwd,
      bytes: options.bytes,
      turnsPerSession: options.turns,
      maxSessions: options.sessions,
      maxMessageBytes: options.maxMessageBytes,
      includeCurrent: options.includeCurrent === true,
      includeSdkCli: options.includeSdkCli === true,
      cwdOnly: options.cwdOnly === true,
    });

    if (options.json) {
      process.stdout.write(JSON.stringify(result, null, 2) + '\n');
      return;
    }

    if (!result.emitted) {
      process.stderr.write(`No recent sessions found for cwd ${result.cwd}.\n`);
      process.exit(1);
    }

    process.stdout.write(result.text);
    const repoList = result.neighborhood.length > 1
      ? ` across ${result.neighborhood.length} repo(s) [${result.neighborhood.map((n) => n.name).join(', ')}]`
      : '';
    const trail = `\n# whisper: ${result.sessionsIncluded} session(s), ${result.turnsIncluded} message(s), ${result.bytes} bytes${result.truncated ? ' (truncated by budget)' : ''}${repoList}\n`;
    process.stderr.write(trail);
  });

// ============================================================================
// gossip - Casual LLM compaction over a whisper (neighborhood paragraph)
// ============================================================================

program
  .command('gossip')
  .description('Streamed prose paragraph over recent agent sessions across the cwd and its conceptual sibling repos. Casual register — like a colleague catching you up over coffee. One cheap LLM pass over a `whisper`, no persistence.')
  .option('--bytes <n>', 'Byte budget for the verbatim window fed to the LLM (default 4000)', (v) => parseInt(v, 10))
  .option('--turns <n>', 'Last N messages per session in the window (default 4)', (v) => parseInt(v, 10))
  .option('--sessions <n>', 'Max sessions to include per repo (default: 5 for self, 2 for siblings)', (v) => parseInt(v, 10))
  .option('--include-current', 'Include the actively-written session (default skipped)')
  .option('--cwd <path>', 'Override current working directory')
  .option('--cwd-only', 'Skip neighborhood expansion — gossip only this cwd (no sibling repos)')
  .option('--model <model>', 'Override the LLM model (default google/gemini-3-flash-preview)')
  .option('--api-key <key>', 'Override the LLM API key')
  .option('--json', 'Emit JSON with the source whisper + paragraph instead of streaming the paragraph to stdout')
  .action(async (options) => {
    if (!hasLlmAccess(options.apiKey)) {
      logger.error('No LLM access. Set an API key (OPENROUTER_API_KEY / "prose config set openrouter-api-key <key>") or a local endpoint ("prose config set llm-base-url http://127.0.0.1:1234/v1")');
      process.exit(1);
    }
    const apiKey = options.apiKey || getLlmConfig().apiKey || 'prose-local';

    // For JSON mode, capture the LLM stream into a buffer instead of stdout
    // so the caller gets a clean structured payload.
    const { Writable } = await import('stream');
    let captured = '';
    const sink = options.json
      ? new Writable({ write(chunk, _enc, cb) { captured += chunk.toString(); cb(); } })
      : process.stdout;

    const result = await gossip({
      apiKey,
      model: options.model,
      cwd: options.cwd,
      bytes: options.bytes,
      turnsPerSession: options.turns,
      maxSessions: options.sessions,
      includeCurrent: options.includeCurrent === true,
      cwdOnly: options.cwdOnly === true,
      out: sink,
    });

    if (options.json) {
      process.stdout.write(JSON.stringify(result, null, 2) + '\n');
      return;
    }

    if (!result.emitted) {
      process.stderr.write(`No recent sessions found for cwd ${result.source.cwd}.\n`);
      process.exit(1);
    }

    const meta = result.source;
    const repoList = meta.neighborhood.length > 1
      ? ` across ${meta.neighborhood.length} repo(s) [${meta.neighborhood.map((n) => n.name).join(', ')}]`
      : '';
    const trail = `\n# gossip: ${meta.sessionsIncluded} session(s), ${meta.turnsIncluded} message(s), ${meta.bytes} bytes in${meta.truncated ? ' (truncated by budget)' : ''}${repoList}\n`;
    process.stderr.write(trail);
  });

// ============================================================================
// standup - Cross-cwd, time-windowed compression of recent activity
// ============================================================================

program
  .command('standup')
  .description('Streamed cross-project standup over recent agent activity (Claude Code CLI, ACP, Codex, opencode, Cursor, pi, and OMP), grouped by working directory')
  .option('--since <duration>', 'Time window for session inclusion: e.g. 30m, 4h, 1d, 2h30m (default 7d). Window decides which sessions to surface; per-session tail length is independent.')
  .option('--turns <n>', "Last N messages per session — taken from the session's overall tail, not the in-window slice (default 10)", (v) => parseInt(v, 10))
  .option('--bytes-per-session <n>', 'Per-session byte cap on rendered tail (default 1500)', (v) => parseInt(v, 10))
  .option('--total-bytes <n>', 'Total byte cap on assembled LLM input (default 2000000)', (v) => parseInt(v, 10))
  .option('--sessions <n>', 'Max sessions across all projects (default 80)', (v) => parseInt(v, 10))
  .option('--include-current', 'Include the actively-written session (default skipped)')
  .option('--include-sdk-cli', 'Include sdk-cli sessions (Claude Code automation: commit-message generators, etc. — noise by default)')
  .option('--model <model>', 'Override the LLM model (default google/gemini-3-flash-preview)')
  .option('--api-key <key>', 'Override the LLM API key')
  .option('--json', 'Emit JSON with metadata + full text instead of streaming to stdout')
  .action(async (options) => {
    if (!hasLlmAccess(options.apiKey)) {
      logger.error('No LLM access. Set an API key (OPENROUTER_API_KEY / "prose config set openrouter-api-key <key>") or a local endpoint ("prose config set llm-base-url http://127.0.0.1:1234/v1")');
      process.exit(1);
    }
    const apiKey = options.apiKey || getLlmConfig().apiKey || 'prose-local';

    const { Writable } = await import('stream');
    let captured = '';
    const sink = options.json
      ? new Writable({ write(chunk, _enc, cb) { captured += chunk.toString(); cb(); } })
      : process.stdout;

    let result;
    try {
      result = await standup({
        apiKey,
        model: options.model,
        since: options.since,
        turnsPerSession: options.turns,
        bytesPerSession: options.bytesPerSession,
        totalBytes: options.totalBytes,
        maxSessions: options.sessions,
        includeCurrent: options.includeCurrent === true,
        includeSdkCli: options.includeSdkCli === true,
        out: sink,
      });
    } catch (err) {
      logger.error(err instanceof Error ? err.message : String(err));
      process.exit(1);
    }

    if (options.json) {
      process.stdout.write(JSON.stringify(result, null, 2) + '\n');
      return;
    }

    if (!result.emitted) {
      process.stderr.write(`No sessions found in the last ${options.since ?? '7d'}.\n`);
      process.exit(1);
    }

    const projectList = result.projects.map(p => `${p.cwd} (${p.sessionCount}s/${p.messageCount}m/${p.commitCount}c)`).join(', ');
    // Headers in the LLM output match the system prompt's "**<short project name>**" format.
    // Counting them keeps the trail honest when the model splits one cwd into multiple
    // project sections (e.g. a 6digit-studio terminal that touched sibling repos).
    const sectionCount = (result.text.match(/^\*\*[^*\n]+\*\*\s*$/gm) ?? []).length;
    const totalCommits = result.projects.reduce((sum, p) => sum + p.commitCount, 0);
    const trail = `\n# standup: ${result.sessionsIncluded} session(s), ${totalCommits} commit(s), ${result.projects.length} cwd(s) in → ${sectionCount} project(s) out, ${result.promptBytes} prompt bytes — ${projectList}\n`;
    process.stderr.write(trail);

    // "17 cwd(s) in → 6 project(s) out" is arithmetic nobody reads as "eleven
    // repos are missing from what you just read". Compaction is allowed to merge
    // and drop; it is not allowed to do it silently. A reader asking "where were
    // we" concludes nothing happened in a repo the report never names — so name
    // the ones that had activity and did not make the prose.
    const unnamed = result.projects.filter(p => {
      const name = p.cwd.split('/').filter(Boolean).pop();
      if (!name) return false;
      // Bounded on both sides so a short name doesn't match a longer sibling
      // ("6digit" must not count itself as covered by "6digit-cordial").
      return !new RegExp(`(^|[^a-z0-9-])${escapeRegex(name)}([^a-z0-9-]|$)`, 'i').test(result.text);
    });
    if (unnamed.length > 0) {
      const list = unnamed
        .map(p => `${p.cwd.split('/').filter(Boolean).pop()} (${p.commitCount}c/${p.sessionCount}s)`)
        .join(', ');
      process.stderr.write(
        `# not named above — had activity in the window, did not make the summary: ${list}\n`
      );
    }
  });

// ============================================================================
// grep - Regex search across the parsed session line-stream (no LLM)
// ============================================================================

program
  .command('grep <pattern...>')
  .description('Regex search across recent agent session text (Claude Code CLI, ACP, Codex, opencode, Cursor, pi, OMP, Devin, and Perplexity). Operates on parsed session content — NOT files on disk. Multiple patterns OR-alternate. Output is grep-style with line numbers and ±N context lines.')
  .option('-C, --context <n>', 'Lines before and after each match (default 5)', (v) => parseInt(v, 10))
  .option('-A, --after <n>', 'Lines after each match (overrides --context for after)', (v) => parseInt(v, 10))
  .option('-B, --before <n>', 'Lines before each match (overrides --context for before)', (v) => parseInt(v, 10))
  .option('-i, --ignore-case', 'Case-insensitive matching')
  .option('-F, --fixed-strings', 'Treat patterns as literal strings, not regex')
  .option('-m, --max-matches <n>', 'Cap on total matches across all sessions (default 50)', (v) => parseInt(v, 10))
  .option('--max-sessions <n>', 'Cap on sessions scanned (performance guardrail, default 500)', (v) => parseInt(v, 10))
  .option('--source <type>', 'Restrict to a source: claude-code | codex | opencode | cursor | pi | omp | devin | perplexity (repeatable)', (v: string, prev: string[] = []) => [...prev, v])
  .option('--cwd <path>', 'Restrict to one cwd (default: all cwds)')
  .option('--since <duration>', 'Time window for inclusion: e.g. 30m, 4h, 1d, 2h30m (default: all time)')
  .option('--include-current', 'Include the actively-written claude-code session')
  .option('--include-sdk-cli', 'Include sdk-cli sessions (Claude Code automation)')
  .option('--json', 'Emit JSON with structured matches instead of text')
  .action((patterns: string[], options) => {
    let sinceMs: number | undefined;
    if (options.since) {
      try {
        sinceMs = parseDuration(options.since);
      } catch (err) {
        logger.error(err instanceof Error ? err.message : String(err));
        process.exit(1);
      }
    }

    let sources: SourceType[] | undefined;
    if (options.source && Array.isArray(options.source)) {
      const valid: SourceType[] = ['claude-code', 'codex', 'opencode', 'cursor', 'pi', 'omp', 'devin', 'perplexity'];
      const bad = options.source.filter((s: string) => !valid.includes(s as SourceType));
      if (bad.length > 0) {
        logger.error(`Unknown --source value(s): ${bad.join(', ')}. Valid: ${valid.join(', ')}.`);
        process.exit(1);
      }
      sources = options.source as SourceType[];
    }

    let result;
    try {
      result = grep({
        patterns,
        cwd: options.cwd,
        sources,
        fixedStrings: options.fixedStrings === true,
        ignoreCase: options.ignoreCase === true,
        context: options.context,
        before: options.before,
        after: options.after,
        sinceMs,
        maxMatches: options.maxMatches,
        maxSessions: options.maxSessions,
        includeCurrent: options.includeCurrent === true,
        includeSdkCli: options.includeSdkCli === true,
      });
    } catch (err) {
      logger.error(err instanceof Error ? err.message : String(err));
      process.exit(1);
    }

    if (options.json) {
      process.stdout.write(JSON.stringify(result, null, 2) + '\n');
      return;
    }

    if (result.matches.length === 0) {
      process.stderr.write(`No matches for ${JSON.stringify(patterns)} across ${result.sessionsScanned} session(s).\n`);
      process.exit(1);
    }

    process.stdout.write(result.text);
    const trail = `\n# grep: ${result.matches.length} match group(s), ${result.totalMatchesBeforeCap} match line(s), ${result.sessionsWithMatches}/${result.sessionsScanned} session(s)${result.truncated ? ' (truncated by cap)' : ''}\n`;
    process.stderr.write(trail);
  });

// ============================================================================
// stats - Quantitative readout over the parsed session stream (no LLM)
// ============================================================================

program
  .command('stats')
  .description('Per-day activity metrics across all agent sessions (Claude Code CLI, ACP, Codex, opencode, Cursor, pi, OMP, Devin, Perplexity): active hours, message volumes, session/project counts, hour-of-day histogram. Global by default — all cwds. Active time merges message timestamps with an idle-gap cutoff; "human" counts user messages only.')
  .option('--since <duration>', 'Time window for inclusion: e.g. 4h, 7d, 2h30m (default 30d)')
  .option('--idle-gap <duration>', 'Gap above which activity splits into separate intervals (default 15m)')
  .option('--source <type>', 'Restrict to a source: claude-code | codex | opencode | cursor | pi | omp | devin | perplexity (repeatable)', (v: string, prev: string[] = []) => [...prev, v])
  .option('--cwd <path>', 'Restrict to one cwd (default: all cwds)')
  .option('--max-sessions <n>', 'Cap on sessions parsed (performance guardrail, default 2000)', (v) => parseInt(v, 10))
  .option('--include-sdk-cli', 'Include sdk-cli sessions (Claude Code automation)')
  .option('--no-cache', 'Bypass the per-file stamp cache (OS temp dir) and re-parse every session file')
  .option('--json', 'Emit full JSON (day buckets, project list, totals, histogram) instead of the table')
  .option('--csv', 'Emit day buckets as CSV for external graphing')
  .action((options) => {
    let sinceMs: number | undefined;
    let idleGapMs: number | undefined;
    try {
      if (options.since) sinceMs = parseDuration(options.since);
      if (options.idleGap) idleGapMs = parseDuration(options.idleGap);
    } catch (err) {
      logger.error(err instanceof Error ? err.message : String(err));
      process.exit(1);
    }

    let sources: SourceType[] | undefined;
    if (options.source && Array.isArray(options.source)) {
      const valid: SourceType[] = ['claude-code', 'codex', 'opencode', 'cursor', 'pi', 'omp', 'devin', 'perplexity'];
      const bad = options.source.filter((s: string) => !valid.includes(s as SourceType));
      if (bad.length > 0) {
        logger.error(`Unknown --source value(s): ${bad.join(', ')}. Valid: ${valid.join(', ')}.`);
        process.exit(1);
      }
      sources = options.source as SourceType[];
    }

    const result = stats({
      cwd: options.cwd,
      sources,
      sinceMs,
      idleGapMs,
      maxSessions: options.maxSessions,
      includeSdkCli: options.includeSdkCli === true,
      cache: options.cache !== false, // commander: --no-cache sets options.cache = false
    });

    if (options.json) {
      process.stdout.write(JSON.stringify(result, null, 2) + '\n');
      return;
    }

    if (result.days.length === 0) {
      process.stderr.write(`No session activity found in the last ${options.since ?? '30d'}.\n`);
      process.exit(1);
    }

    if (options.csv) {
      process.stdout.write(renderCsv(result.days));
      return;
    }

    process.stdout.write(result.text);
    const trail = `\n# stats: ${result.totals.activeDays} day(s), ${result.sessionsScanned} session(s) scanned, ${result.cacheHits} file(s) from cache${result.truncated ? ' (truncated by --max-sessions)' : ''}\n`;
    process.stderr.write(trail);
  });

// ============================================================================
// session - Verbatim readout of a single session by id (or id prefix)
// ============================================================================

program
  .command('session <id>')
  .description('Verbatim readout of one session by id (accepts any unique prefix). Scans Claude Code CLI, ACP, Codex, opencode, Cursor, pi, and OMP. No cwd filter — the id is the selector.')
  .option('--turns <n>', 'Tail the last N messages (default: all)', (v) => parseInt(v, 10))
  .option('--since <iso>', 'Only include messages at or after this ISO timestamp')
  .option('--max-message-bytes <n>', 'Per-message byte cap; long messages get truncated with [N bytes elided] (default 0 = no clipping)', (v) => parseInt(v, 10))
  .option('--no-sdk-cli', 'Refuse to match a Claude-Code sdk-cli automation session (default: allow)')
  .option('--json', 'Emit JSON with metadata + text instead of plain text')
  .action((id: string, options) => {
    let since: Date | undefined;
    if (options.since) {
      const parsed = new Date(options.since);
      if (Number.isNaN(parsed.getTime())) {
        logger.error(`Invalid --since timestamp: ${options.since}`);
        process.exit(1);
      }
      since = parsed;
    }

    let result;
    try {
      result = session(id, {
        turns: options.turns,
        since,
        maxMessageBytes: options.maxMessageBytes,
        includeSdkCli: options.sdkCli !== false,
      });
    } catch (err) {
      if (err instanceof SessionAmbiguousError) {
        process.stderr.write(err.message + '\n');
        process.exit(2);
      }
      if (err instanceof SessionNotFoundError) {
        process.stderr.write(err.message + '\n');
        process.exit(1);
      }
      logger.error(err instanceof Error ? err.message : String(err));
      process.exit(1);
    }

    if (options.json) {
      process.stdout.write(JSON.stringify(result, null, 2) + '\n');
      return;
    }

    process.stdout.write(result.text);
    const trail = `\n# session: ${result.messagesIncluded}/${result.messageCount} message(s), ${result.bytes} bytes, ${result.sourceType}\n`;
    process.stderr.write(trail);
  });

// ============================================================================
// tail - Follow a live session as it grows
// ============================================================================

program
  .command('tail [id]')
  .description('Follow a live session as it grows — verbatim, no LLM. With an id prefix, pins that session; with none, picks the most recent session for the cwd (--cwd to point elsewhere, --any for globally newest). Initial backlog then each new message as it lands. The observation deck for watching another agent work.')
  .option('--turns <n>', 'Initial backlog: last N messages before following (default 10)', (v) => parseInt(v, 10))
  .option('--interval <ms>', 'Poll interval in milliseconds (default 2000)', (v) => parseInt(v, 10))
  .option('--max-message-bytes <n>', 'Per-message byte cap (default 4096; 0 = full firehose)', (v) => parseInt(v, 10))
  .option('--cwd <path>', 'With no id: pick the newest session for this cwd (default: current directory)')
  .option('--any', 'With no id: pick the globally newest session, any cwd')
  .action(async (id: string | undefined, options) => {
    try {
      await tail(id, {
        turns: options.turns,
        intervalMs: options.interval,
        maxMessageBytes: options.maxMessageBytes,
        cwd: options.cwd,
        any: options.any,
      });
    } catch (err) {
      if (err instanceof SessionAmbiguousError || err instanceof SessionNotFoundError || err instanceof NoSessionForCwdError) {
        process.stderr.write(err.message + '\n');
        process.exit(err instanceof SessionAmbiguousError ? 2 : 1);
      }
      logger.error(err instanceof Error ? err.message : String(err));
      process.exit(1);
    }
  });

// ============================================================================
// skill - Install/uninstall the prose Claude Code skill
// ============================================================================

const skillCmd = program
  .command('skill')
  .description('Install or manage the `prose` Claude Code skill — drops a SKILL.md into ~/.claude/skills/prose/ so future Claude sessions know how and when to use prose');

skillCmd
  .command('install')
  .description('Install the prose skill into ~/.claude/skills/prose/SKILL.md')
  .option('--force', 'Overwrite an existing SKILL.md')
  .action(async (options) => {
    const { fileURLToPath } = await import('url');
    const { dirname, join } = await import('path');
    const fs = await import('fs');
    const os = await import('os');

    const here = dirname(fileURLToPath(import.meta.url));
    // dist/cli.js → ../skill/SKILL.md
    const sourcePath = join(here, '..', 'skill', 'SKILL.md');
    if (!fs.existsSync(sourcePath)) {
      logger.error(`Skill source missing at ${sourcePath}. This is a packaging bug — file an issue.`);
      process.exit(1);
    }

    const targetDir = join(os.homedir(), '.claude', 'skills', 'prose');
    const targetPath = join(targetDir, 'SKILL.md');

    if (fs.existsSync(targetPath) && !options.force) {
      logger.warn(`SKILL.md already exists at ${targetPath}.`);
      logger.warn('Use --force to overwrite, or `prose skill uninstall` first.');
      process.exit(1);
    }

    fs.mkdirSync(targetDir, { recursive: true });
    fs.copyFileSync(sourcePath, targetPath);

    logger.success(`Installed prose skill at ${targetPath}`);
    logger.info('Future Claude Code sessions will pick it up automatically.');
  });

skillCmd
  .command('uninstall')
  .description('Remove ~/.claude/skills/prose/SKILL.md')
  .action(async () => {
    const { join } = await import('path');
    const fs = await import('fs');
    const os = await import('os');

    const targetDir = join(os.homedir(), '.claude', 'skills', 'prose');
    const targetPath = join(targetDir, 'SKILL.md');

    if (!fs.existsSync(targetPath)) {
      logger.warn(`No prose skill installed at ${targetPath}. Nothing to do.`);
      return;
    }

    fs.unlinkSync(targetPath);
    // Remove the directory if it's empty.
    try {
      fs.rmdirSync(targetDir);
    } catch {
      // Directory not empty (user may have added other files) — leave it alone.
    }

    logger.success(`Removed prose skill from ${targetPath}`);
  });

skillCmd
  .command('show')
  .description('Print the prose SKILL.md contents to stdout (so you can preview before installing)')
  .action(async () => {
    const { fileURLToPath } = await import('url');
    const { dirname, join } = await import('path');
    const fs = await import('fs');

    const here = dirname(fileURLToPath(import.meta.url));
    const sourcePath = join(here, '..', 'skill', 'SKILL.md');
    if (!fs.existsSync(sourcePath)) {
      logger.error(`Skill source missing at ${sourcePath}.`);
      process.exit(1);
    }

    process.stdout.write(fs.readFileSync(sourcePath, 'utf-8'));
  });

skillCmd
  .command('path')
  .description('Print the install path (~/.claude/skills/prose/SKILL.md) without installing')
  .action(async () => {
    const { join } = await import('path');
    const os = await import('os');
    process.stdout.write(join(os.homedir(), '.claude', 'skills', 'prose', 'SKILL.md') + '\n');
  });

// ============================================================================
// config - Manage global configuration
// ============================================================================

// ============================================================================
// baton - prose-native "you are here" markers (the one stateful verb)
// ============================================================================
function formatBatonAge(ms: number): string {
  if (ms < 60_000) return `${Math.round(ms / 1000)}s ago`;
  if (ms < 3_600_000) return `${Math.round(ms / 60_000)}m ago`;
  if (ms < 86_400_000) return `${Math.round(ms / 3_600_000)}h ago`;
  return `${Math.round(ms / 86_400_000)}d ago`;
}

const batonCmd = program
  .command('baton')
  .description('Persisted "you are here" batons. Bare `prose baton` lists the latest per project/type.');

batonCmd
  .command('set [content...]')
  .description('Write a baton for the cwd. Accepts a `↪ LABEL: body` form, or set the label with --type. With no content (or `-`), reads the body from stdin — use that for long batons, since a shell will otherwise eat backticks and $() inside them.')
  .option('--type <label>', 'Baton type label (default: parsed from content, else "baton")')
  .option('--cwd <path>', 'Project the baton belongs to (default: current directory)')
  .option('--json', 'Emit the stored baton as JSON')
  .action((content: string[] | undefined, options) => {
    try {
      const argv = content ?? [];
      // A baton is a hand-written note whose whole value is being preserved
      // verbatim. Passing one as shell argv silently corrupts it the moment it
      // contains a backtick or $( — so stdin is the safe channel, and it is
      // what you get by default when no content is on the command line.
      const fromStdin = argv.length === 0 || (argv.length === 1 && argv[0] === '-');
      let body: string;
      if (fromStdin) {
        if (process.stdin.isTTY) {
          logger.error('baton set: no content given and stdin is a terminal — pass the body as arguments, or pipe it in');
          process.exit(1);
        }
        body = readFileSync(0, 'utf8').trim();
        if (!body) {
          logger.error('baton set: stdin was empty');
          process.exit(1);
        }
      } else {
        body = argv.join(' ');
      }
      const baton = setBaton({
        content: body,
        type: options.type,
        cwd: options.cwd,
      });
      if (options.json) {
        process.stdout.write(JSON.stringify(baton, null, 2) + '\n');
        return;
      }
      logger.success(`Baton set for ${baton.project}`);
      process.stdout.write(renderBatonLine(baton) + '\n');
    } catch (err) {
      logger.error(err instanceof Error ? err.message : String(err));
      process.exit(1);
    }
  });

batonCmd
  .command('list', { isDefault: true })
  .description('Show batons — latest per type for the current project by default (--global for all).')
  .option('--cwd <path>', 'Scope to a specific project (default: the current directory)')
  .option('--global', 'Show batons across every project, not just this one')
  .option('--type <label>', 'Filter to one type')
  .option('--history', 'Show full history instead of latest-per-project/type')
  .option('--limit <n>', 'Cap the number shown', (v) => parseInt(v, 10))
  .option('--json', 'Emit JSON')
  .action((options) => {
    // Default to the current project — a baton is a "you are here," and you
    // almost always want *this* garden's, the same way `snap` defaults to cwd.
    // `--global` opts into the cross-project view; an explicit `--cwd` wins.
    const cwd = options.global ? undefined : (options.cwd ?? process.cwd());
    const batons = listBatons({
      cwd,
      type: options.type,
      history: options.history === true,
      limit: options.limit,
    });
    if (options.json) {
      process.stdout.write(JSON.stringify(batons, null, 2) + '\n');
      return;
    }
    if (batons.length === 0) {
      process.stderr.write('No batons yet.\n');
      return;
    }
    const now = Date.now();
    for (const b of batons) {
      const age = formatBatonAge(now - new Date(b.timestamp).getTime());
      const origin = batonOriginNote(b);
      const originLine = origin ? `  ·  ${origin}` : '';
      process.stdout.write(`${renderBatonLine(b)}\n    ${b.project}  ·  ${age}${originLine}\n`);
    }
  });

batonCmd
  .command('clear')
  .description('Remove batons. Scope with --cwd/--type, or --all to wipe everything.')
  .option('--cwd <path>', 'Only clear batons for this project')
  .option('--type <label>', 'Only clear batons of this type')
  .option('--all', 'Clear every baton (required for an unscoped wipe)')
  .action((options) => {
    try {
      const removed = clearBatons({
        cwd: options.cwd,
        type: options.type,
        all: options.all === true,
      });
      logger.success(`Removed ${removed} baton(s).`);
    } catch (err) {
      logger.error(err instanceof Error ? err.message : String(err));
      process.exit(1);
    }
  });

const configCmd = program.command('config').description('Manage global configuration settings');

configCmd
  .command('set <key> <value...>')
  .description('Set a global configuration value (use space-separated values for arrays)')
  .action((key, values) => {
    const value = values.length === 1 ? values[0] : values;

    if (key === 'artifacts') {
      saveGlobalConfig({ artifacts: value === 'true' });
      logger.success(`Set artifacts to: ${value === 'true'}`);
    } else if (key === 'mirror-mode') {
      if (value !== 'vault' && value !== 'local') {
        logger.error('mirror-mode must be "vault" or "local"');
        process.exit(1);
      }
      saveGlobalConfig({ mirrorMode: value as any });
      logger.success(`Set mirror-mode to: ${value}`);
    } else if (key === 'source-extensions') {
      // Accept space-separated extensions: prose config set source-extensions .ts .js .svelte
      const extensions = Array.isArray(value) ? value : [value];
      // Ensure each extension starts with a dot
      const normalized = extensions.map(ext => ext.startsWith('.') ? ext : `.${ext}`);
      saveGlobalConfig({ sourceExtensions: normalized });
      logger.success(`Set source-extensions to: ${normalized.join(', ')}`);
    } else if (key === 'auto-index-source') {
      saveGlobalConfig({ autoIndexSource: value === 'true' });
      logger.success(`Set auto-index-source to: ${value === 'true'}`);
    } else if (key === 'vector-threshold') {
      const threshold = parseFloat(value as string);
      if (isNaN(threshold) || threshold < 0 || threshold > 1) {
        logger.error('vector-threshold must be a number between 0.0 and 1.0');
        process.exit(1);
      }
      saveGlobalConfig({ vectorThreshold: threshold });
      logger.success(`Set vector-threshold to: ${threshold}`);
    } else if (key === 'jina-api-key') {
      saveGlobalConfig({ jinaApiKey: value as string });
      logger.success(`Set jina-api-key (${(value as string).length} chars)`);
    } else if (key === 'openrouter-api-key') {
      saveGlobalConfig({ openRouterApiKey: value as string });
      logger.success(`Set openrouter-api-key (${(value as string).length} chars)`);
    } else if (key === 'llm-api-key') {
      saveGlobalConfig({ llmApiKey: value as string });
      logger.success(`Set llm-api-key (${(value as string).length} chars)`);
    } else if (key === 'llm-base-url') {
      saveGlobalConfig({ llmBaseUrl: value as string });
      logger.success(`Set llm-base-url to: ${value}`);
    } else if (key === 'llm-model') {
      saveGlobalConfig({ llmModel: value as string });
      logger.success(`Set llm-model to: ${value}`);
    } else {
      logger.error(`Unknown configuration key: ${key}`);
      logger.info('Valid keys: artifacts, mirror-mode, source-extensions, auto-index-source, vector-threshold, jina-api-key, openrouter-api-key, llm-api-key, llm-base-url, llm-model');
      process.exit(1);
    }
  });

configCmd
  .command('show')
  .description('Show current global configuration')
  .action(() => {
    const config = getGlobalConfig();
    const maskKey = (key: string | undefined) => {
      if (!key) return '(not set)';
      if (key.length <= 8) return '****';
      return key.slice(0, 4) + '...' + key.slice(-4);
    };

    logger.info('📊 Prose - Global Configuration\n');
    console.log('   General:');
    console.log(`     artifacts: ${config.artifacts}`);
    console.log(`     mirror-mode: ${config.mirrorMode}`);
    console.log('\n   Source Indexing:');
    console.log(`     source-extensions: ${config.sourceExtensions?.join(', ') || '(none)'}`);
    console.log(`     auto-index-source: ${config.autoIndexSource}`);
    console.log(`     vector-threshold: ${config.vectorThreshold}`);
    console.log('\n   LLM Endpoint:');
    const llm = getLlmConfig();
    console.log(`     effective base-url: ${llm.baseUrl || 'https://openrouter.ai/api/v1 (default)'}`);
    console.log(`     effective model: ${llm.model || 'google/gemini-3-flash-preview (default)'}`);
    console.log('\n   API Keys:');
    console.log(`     jina-api-key: ${maskKey(config.jinaApiKey)}`);
    console.log(`     openrouter-api-key: ${maskKey(config.openRouterApiKey)}`);
    console.log(`     llm-api-key: ${maskKey(config.llmApiKey)}`);
    console.log(`\n📂 Vault location: ${getMemoryDir()}`);
  });

// ============================================================================
// vault - Manage the personal memory storage repository
// ============================================================================

const vault = program.command('vault').description('Manage the personal memory vault (Git-backed storage)');

vault
  .command('init [remote]')
  .description('Initialize the memory directory as a Git repository')
  .action(async (remote) => {
    const memoryDir = getMemoryDir();
    if (!existsSync(memoryDir)) {
      mkdirSync(memoryDir, { recursive: true });
    }

    if (isVaultRepo()) {
      logger.info('🏛️  Vault already initialized.');
    } else {
      try {
        const { execSync } = await import('child_process');
        execSync(`git -C "${memoryDir}" init`, { stdio: 'inherit' });

        // Add a .gitignore - exclude temp files and derived data (vectors)
        const gitignorePath = join(memoryDir, '.gitignore');
        if (!existsSync(gitignorePath)) {
          writeFileSync(gitignorePath, `# Temporary files
*.tmp
*.bak

# Derived data - regenerate with: prose index backfill
*.vectors.json
*.source-vectors.json
*.source.json
`);
        }

        execSync(`git -C "${memoryDir}" add .`, { stdio: 'inherit' });
        execSync(`git -C "${memoryDir}" commit -m "Initialize Personal Memory Vault"`, { stdio: 'inherit' });
        logger.info('✅ Vault initialized successfully.');
      } catch (error: any) {
        logger.error(`⚠️  Failed to initialize vault: ${error.message}`);
        return;
      }
    }

    if (remote) {
      try {
        const { execSync } = await import('child_process');
        execSync(`git -C "${memoryDir}" remote add origin "${remote}"`, { stdio: 'inherit' });
        logger.info(`🔗 Remote added: ${remote}`);
      } catch (error: any) {
        logger.error(`⚠️  Failed to add remote: ${error.message}`);
      }
    }
  });

vault
  .command('status')
  .description('Show the status of the memory vault')
  .action(async () => {
    if (!isVaultRepo()) {
      logger.error('❌ Vault not initialized. Run "prose vault init" first.');
      return;
    }

    try {
      const { execSync } = await import('child_process');
      const memoryDir = getMemoryDir();
      const output = execSync(`git -C "${memoryDir}" status`, { encoding: 'utf-8' });
      console.log(output);
    } catch (error: any) {
      logger.error(`⚠️  Failed to get vault status: ${error.message}`);
    }
  });

vault
  .command('sync')
  .description('Synchronize the vault with the remote repository')
  .action(async () => {
    if (!isVaultRepo()) {
      logger.error('❌ Vault not initialized. Run "prose vault init" first.');
      return;
    }

    try {
      const { execSync } = await import('child_process');
      const memoryDir = getMemoryDir();

      logger.info('🔄 Syncing vault...');

      // 1. Detect current branch
      const currentBranch = execSync(`git -C "${memoryDir}" rev-parse --abbrev-ref HEAD`, { encoding: 'utf-8' }).trim();

      // 2. Check if remote has this branch
      const remoteRefs = execSync(`git -C "${memoryDir}" ls-remote origin ${currentBranch}`, { encoding: 'utf-8' }).trim();

      if (remoteRefs) {
        // Branch exists on remote, try to pull
        logger.trace(`Pulling ${currentBranch} from origin...`);
        execSync(`git -C "${memoryDir}" pull --rebase origin ${currentBranch}`, { stdio: 'inherit' });
      } else {
        logger.info(`✨ First sync for branch "${currentBranch}" - skipping pull.`);
      }

      // 3. Push current branch
      logger.trace(`Pushing ${currentBranch} to origin...`);
      execSync(`git -C "${memoryDir}" push origin ${currentBranch}`, { stdio: 'inherit' });

      logger.info('✅ Vault synchronized.');
    } catch (error: any) {
      logger.error(`⚠️  Sync failed: ${error.message}`);
    }
  });

// ============================================================================
// index - Manage semantic search vectors
// ============================================================================

const indexCmd = program.command('index').description('Manage semantic search vectors');

indexCmd
  .command('backfill')
  .description('Generate missing embeddings for all existing project fragments')
  .option('-p, --project <name>', 'Filter to specific project')
  .action(async (options) => {
    const apiKey = getApiKey('jina');
    if (!apiKey) {
      logger.error('No Jina API key found. Set PROSE_JINA_API_KEY env var or use "prose config set jina-api-key <key>"');
      process.exit(1);
    }

    const index = loadMemoryIndex();
    const projectsToProcess = options.project
      ? Object.keys(index.projects).filter(p => p === options.project)
      : Object.keys(index.projects);

    if (projectsToProcess.length === 0) {
      logger.info('No projects found to backfill.');
      return;
    }

    for (const projectName of projectsToProcess) {
      logger.info(`🧠 Backfilling vectors for ${projectName}...`);
      const memory = loadProjectMemory(projectName);
      if (!memory) continue;

      const vectors = loadProjectVectors(projectName);
      const toEmbed: { hash: string; text: string }[] = [];

      // Helper to collect fragments from an AllFragments object
      const collectFragments = (fragments: any) => {
        // Decisions
        for (const d of fragments.decisions?.decisions || []) {
          const hash = calculateFragmentHash('decision', d.what, d.why);
          if (!vectors[hash]) toEmbed.push({ hash, text: `${d.what} ${d.why}` });
        }
        // Insights
        for (const i of fragments.insights?.insights || []) {
          const hash = calculateFragmentHash('insight', i.learning, i.context);
          if (!vectors[hash]) toEmbed.push({ hash, text: `${i.learning} ${i.context}` });
        }
        // Gotchas
        for (const g of fragments.insights?.gotchas || []) {
          const hash = calculateFragmentHash('gotcha', g.issue, g.solution);
          if (!vectors[hash]) toEmbed.push({ hash, text: `${g.issue} ${g.solution}` });
        }
        // Story beats
        for (const b of fragments.narrative?.story_beats || []) {
          const hash = calculateFragmentHash('narrative', b.summary, b.beat_type);
          if (!vectors[hash]) toEmbed.push({ hash, text: b.summary });
        }
        // Quotes
        for (const q of fragments.narrative?.memorable_quotes || []) {
          const hash = calculateFragmentHash('quote', q.quote, q.speaker);
          if (!vectors[hash]) toEmbed.push({ hash, text: `${q.quote} - ${q.speaker}` });
        }
      };

      // Collect from session snapshots (historical)
      for (const snapshot of memory.sessionSnapshots) {
        collectFragments(snapshot.fragments);
      }

      // Collect from current (evolved/merged) - these are the ground truth
      if (memory.current) {
        collectFragments(memory.current);
      }

      if (toEmbed.length === 0) {
        logger.info(`✅ ${projectName} is already up to date.`);
        continue;
      }

      logger.info(`📡 Requesting ${toEmbed.length} embeddings from Jina...`);
      try {
        // Process in batches of 50 to avoid Jina API limits
        const batchSize = 50;
        for (let i = 0; i < toEmbed.length; i += batchSize) {
          const batch = toEmbed.slice(i, i + batchSize);
          const embeddings = await getJinaEmbeddings(batch.map(b => b.text), apiKey);
          batch.forEach((item, index) => {
            vectors[item.hash] = embeddings[index];
          });
          logger.info(`   Progress: ${Math.min(i + batchSize, toEmbed.length)}/${toEmbed.length}`);
        }
        saveProjectVectors(projectName, vectors);
        logger.success(`✅ ${projectName} backfill complete.`);
      } catch (error: any) {
        logger.error(`❌ Failed to backfill ${projectName}: ${error.message}`);
      }
    }
  });

indexCmd
  .command('source')
  .description('Index project source code for semantic implementation search')
  .option('-p, --project <name>', 'Project name or path')
  .option('-a, --all', 'Index all known projects (backfill)')
  .action(async (options) => {
    const apiKey = getApiKey('jina');
    if (!apiKey) {
      logger.error('No Jina API key found. Set PROSE_JINA_API_KEY env var or use "prose config set jina-api-key <key>"');
      process.exit(1);
    }

    // Backfill all projects
    if (options.all) {
      const index = loadMemoryIndex();
      const projects = Object.keys(index.projects);

      logger.info(`🔍 Backfilling source index for ${projects.length} projects...\n`);

      let totalFiles = 0;
      let totalChunks = 0;
      let indexed = 0;
      let skipped = 0;

      for (const projectName of projects) {
        const memory = loadProjectMemory(projectName);
        const rootPath = memory?.rootPath;

        if (!rootPath || !existsSync(rootPath)) {
          logger.verbose(`⏭️  ${formatProjectName(projectName)}: no rootPath or path doesn't exist`);
          skipped++;
          continue;
        }

        if (!isGitRepo(rootPath)) {
          logger.verbose(`⏭️  ${formatProjectName(projectName)}: not a git repo`);
          skipped++;
          continue;
        }

        try {
          logger.info(`📂 ${formatProjectName(projectName)}`);
          const stats = await indexProjectSource(projectName, rootPath, apiKey);
          logger.info(`   ✅ ${stats.filesIndexed} files, ${stats.chunksCreated} new chunks`);
          totalFiles += stats.filesIndexed;
          totalChunks += stats.chunksCreated;
          indexed++;
        } catch (error: any) {
          logger.warn(`   ⚠️  Failed: ${error.message}`);
        }
      }

      logger.info(`\n📊 Summary:`);
      logger.info(`   Projects indexed: ${indexed}`);
      logger.info(`   Projects skipped: ${skipped}`);
      logger.info(`   Total files: ${totalFiles}`);
      logger.info(`   Total new chunks: ${totalChunks}`);
      return;
    }

    // Single project mode
    let projectName = options.project;
    if (!projectName) {
      projectName = detectProjectFromCwd();
      if (!projectName) {
        logger.error('Could not detect project from current directory');
        process.exit(1);
      }
    }

    try {
      const stats = await indexProjectSource(projectName, process.cwd(), apiKey);
      logger.success(`✅ ${projectName} source indexing complete.`);
      logger.info(`   Files indexed: ${stats.filesIndexed}`);
      logger.info(`   Chunks created: ${stats.chunksCreated}`);
    } catch (error: any) {
      logger.error(`❌ Failed to index source: ${error.message}`);
    }
  });

// ============================================================================
// chronicle - a live dev-feed of freeform beats (durable log + optional sinks)
// ============================================================================

const chronicleCmd = program
  .command('chronicle [title]')
  .description('Record a live dev beat — appends to the durable log and emits to any configured sink')
  .option('--emoji <emoji>', 'Emoji hint for the beat (e.g. 💡 🐛 ✅ 🔥)')
  .option('--body <text>', 'Body text. Falls back to stdin if omitted.')
  .option('--dry-run', 'Print the payload; write nothing, post nothing')
  .action(async (title: string | undefined, opts: { emoji?: string; body?: string; dryRun?: boolean }) => {
    if (!title) {
      chronicleCmd.help();
      return;
    }

    // Body falls back to stdin (matches Koru's post-dev-note.js).
    let body = opts.body;
    if (body == null && !process.stdin.isTTY) {
      let data = '';
      for await (const chunk of process.stdin) data += chunk;
      if (data.trim()) body = data;
    }

    const cwd = process.cwd();
    // Chronicle must work from the first beat in a fresh repo — detectProjectFromCwd
    // returns undefined when there's no evolved memory and no sessions yet, so fall
    // back to the sanitized cwd (same key shape a later `evolve` will land on).
    const project = detectProjectFromCwd() ?? sanitizePath(cwd);
    const config = loadChronicleConfig(cwd);
    const content = buildContent(opts.emoji, title, body);

    if (opts.dryRun) {
      console.log('[DRY RUN] Would record this beat:');
      console.log('---');
      console.log(content);
      console.log('---');
      console.log(`(${content.length} chars, project: ${formatProjectName(project)})`);
      const discord = config?.sinks?.discord;
      console.log(discord?.webhook
        ? `Would emit to Discord${discord.username ? ` as "${discord.username}"` : ''}.`
        : 'No Discord sink configured — would write the durable log only.');
      return;
    }

    // 1. Always append to the durable log first.
    const entry: ChronicleEntry = {
      ts: new Date().toISOString(),
      ...(opts.emoji ? { emoji: opts.emoji } : {}),
      title: title.trim(),
      ...(body?.trim() ? { body: body.trim() } : {}),
    };
    appendChronicleEntry(project, entry);
    logger.success(`📓 Logged beat to ${formatProjectName(project)}`);

    // 2. Emit to the Discord sink if one is configured. No sink is not an error.
    const discord = config?.sinks?.discord;
    if (discord?.webhook) {
      await emitToDiscord(discord, content);
      logger.success(`📡 Posted to Discord${discord.channel ? ` (${discord.channel})` : ''}`);
    }
  });

chronicleCmd
  .command('about')
  .description('Print the charter — what this repo is chronicling, and in what voice')
  .action(() => {
    const config = loadChronicleConfig(process.cwd());
    if (!config) {
      logger.info('No chronicle config yet. Run `prose chronicle init` to scaffold one.');
      return;
    }
    logger.info('📓 Chronicle charter\n');
    console.log(`   about:      ${config.about ?? '(not set)'}`);
    console.log(`   enthusiasm: ${config.enthusiasm ?? '(not set)'}`);
    const discord = config.sinks?.discord;
    console.log(`   discord:    ${discord?.webhook ? `configured${discord.channel ? ` → ${discord.channel}` : ''}` : '(no sink)'}`);
  });

chronicleCmd
  .command('init')
  .description('Scaffold .claude/prose/chronicle.json (charter + enthusiasm + sinks)')
  .action(() => {
    const cwd = process.cwd();
    const path = getChronicleConfigPath(cwd);
    if (existsSync(path)) {
      logger.info(`ℹ️  Chronicle config already exists at ${path} — not overwriting.`);
      return;
    }
    const dir = dirname(path);
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

    // Ensure .claude/prose/ is gitignored BEFORE writing the secret webhook.
    // chronicle init may run standalone (without a prior `prose init`), so it
    // can't assume the ignore entry already exists — otherwise the webhook
    // becomes committable. Mirrors the gitignore logic in `prose init`.
    const proseIgnore = '.claude/prose/';
    const gitignorePath = join(cwd, '.gitignore');
    if (existsSync(gitignorePath)) {
      const gitignoreContent = readFileSync(gitignorePath, 'utf-8');
      if (!gitignoreContent.includes(proseIgnore)) {
        writeFileSync(gitignorePath, `${gitignoreContent}\n# Prose local data (carries the chronicle webhook secret)\n${proseIgnore}\n`);
        logger.info(`🛡️  Added ${proseIgnore} to .gitignore`);
      }
    } else if (isGitRepo(cwd)) {
      writeFileSync(gitignorePath, `# Prose local data (carries the chronicle webhook secret)\n${proseIgnore}\n`);
      logger.info(`🛡️  Created .gitignore with ${proseIgnore}`);
    } else {
      logger.info(`⚠️  Not a git repo — skipping .gitignore. Keep ${proseIgnore} out of version control yourself.`);
    }

    const template = {
      about: 'What this repo is chronicling, and in what voice. e.g. "Live-coding X. Post breakthroughs, ugly bugs, and \'oh shit it compiles\' moments. Voice: terse, a little unhinged, emoji-forward."',
      enthusiasm: 'high',
      sinks: {
        discord: {
          webhook: 'https://discord.com/api/webhooks/...',
          channel: '#dev-notes',
          username: 'your-name-here',
        },
      },
    };
    writeFileSync(path, JSON.stringify(template, null, 2));
    logger.success(`📓 Scaffolded ${path}`);
    logger.info('   Fill in the Discord webhook (this file is gitignored — the webhook is a secret).');
  });

program.parse();
