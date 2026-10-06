/**
 * Shared LLM endpoint resolution.
 *
 * Every LLM call site (evolve, gossip, standup, design) used to hardcode
 * OpenRouter + gemini-3-flash. This module resolves the effective endpoint
 * through one priority chain:
 *
 *   baseUrl:  explicit opt > PROSE_LLM_BASE_URL > config llm-base-url
 *             > Perplexity app's local inference endpoint > OpenRouter
 *   model:    explicit opt > PROSE_LLM_MODEL > config llm-model
 *             > Perplexity app's inference model > gemini-3-flash
 *   apiKey:   explicit opt > getApiKey('llm') chain > Perplexity's key
 *             (only when its baseUrl was adopted) > 'prose-local'
 *
 * A custom baseUrl with no key is legal — local servers (LM Studio etc.)
 * accept any bearer string, so we send a placeholder rather than failing.
 * getLlmConfig() therefore returns baseUrl/model ONLY when explicitly
 * configured; callers apply the OpenRouter/Gemini defaults themselves so
 * "unset" stays distinguishable from "default".
 */

import { getApiKey, getGlobalConfig } from './memory.js';
import { getPerplexityInferenceConfig } from './perplexity-session-parser.js';

export const DEFAULT_LLM_BASE_URL = 'https://openrouter.ai/api/v1';
export const DEFAULT_LLM_MODEL = 'google/gemini-3-flash-preview';

export interface LlmConfig {
  /** Resolved API key, or undefined when none is configured anywhere. */
  apiKey?: string;
  /** Explicitly configured base URL — undefined means "use the default". */
  baseUrl?: string;
  /** Explicitly configured model — undefined means "use the default". */
  model?: string;
}

export function getLlmConfig(): LlmConfig {
  const config = getGlobalConfig();
  const pplx = getPerplexityInferenceConfig();

  const baseUrl =
    process.env.PROSE_LLM_BASE_URL ||
    config.llmBaseUrl ||
    pplx?.baseUrl ||
    undefined;
  const model =
    process.env.PROSE_LLM_MODEL ||
    config.llmModel ||
    pplx?.model ||
    undefined;

  let apiKey = getApiKey('llm');
  if (!apiKey && pplx?.apiKey && baseUrl === pplx.baseUrl) {
    apiKey = pplx.apiKey;
  }

  return { apiKey, baseUrl, model };
}

export interface ResolvedLlmParams {
  apiKey: string;
  baseUrl: string;
  model: string;
}

/**
 * Fully resolved endpoint params for a call site: opts > configured > default.
 * apiKey falls back to a placeholder string — local OpenAI-compatible servers
 * ignore the bearer token, and remote endpoints will 401 with a clearer
 * server-side error than a missing-key crash.
 */
export function resolveLlmParams(opts: {
  apiKey?: string;
  baseUrl?: string;
  model?: string;
} = {}): ResolvedLlmParams {
  const llm = getLlmConfig();
  return {
    apiKey: opts.apiKey || llm.apiKey || 'prose-local',
    baseUrl: opts.baseUrl || llm.baseUrl || DEFAULT_LLM_BASE_URL,
    model: opts.model || llm.model || DEFAULT_LLM_MODEL,
  };
}

/**
 * Can we plausibly reach an LLM? True when any API key exists OR a custom
 * endpoint is configured (local endpoints are commonly keyless). This is what
 * the CLI's `if (!apiKey) abort` guards should ask instead.
 */
export function hasLlmAccess(explicitKey?: string): boolean {
  if (explicitKey) return true;
  const llm = getLlmConfig();
  return Boolean(llm.apiKey || llm.baseUrl);
}
