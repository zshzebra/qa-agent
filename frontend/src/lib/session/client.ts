import { writable, type Readable } from 'svelte/store';

import { consumeEventStream } from './sse';
import type {
	SessionClientOptions,
	SessionSnapshot,
	SessionStatus,
	SessionStoreState
} from './types';

const SESSION_STATUSES: SessionStatus[] = ['working', 'waiting_for_user', 'completed', 'failed'];

export interface SessionClient extends Readable<SessionStoreState> {
	hydrate(): Promise<SessionSnapshot>;
	start(): Promise<void>;
	sendMessage(content: string): Promise<SessionSnapshot>;
	stop(): void;
}

export class SessionApiError extends Error {
	constructor(
		message: string,
		readonly status: number
	) {
		super(message);
		this.name = 'SessionApiError';
	}
}

function isSessionSnapshot(value: unknown): value is SessionSnapshot {
	if (!value || typeof value !== 'object') return false;
	const candidate = value as Partial<SessionSnapshot>;
	return (
		typeof candidate.revision === 'number' &&
		typeof candidate.status === 'string' &&
		SESSION_STATUSES.includes(candidate.status as SessionStatus) &&
		Array.isArray(candidate.messages)
	);
}

function errorMessage(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

async function responseError(response: Response): Promise<SessionApiError> {
	let message = response.statusText || `Request failed with status ${response.status}`;
	const text = await response.text();
	try {
		const body = JSON.parse(text) as { detail?: unknown };
		if (body.detail) message = String(body.detail);
	} catch {
		if (text) message = text;
	}
	return new SessionApiError(message, response.status);
}

function delay(milliseconds: number, signal: AbortSignal): Promise<void> {
	return new Promise((resolve) => {
		const timeout = setTimeout(resolve, milliseconds);
		signal.addEventListener(
			'abort',
			() => {
				clearTimeout(timeout);
				resolve();
			},
			{ once: true }
		);
	});
}

export function createSessionClient(options: SessionClientOptions): SessionClient {
	if (!options.key) throw new Error('A session key is required');

	const baseUrl = (options.baseUrl ?? '/api').replace(/\/+$/, '');
	const fetcher = options.fetch ?? globalThis.fetch.bind(globalThis);
	const reconnectDelayMs = options.reconnectDelayMs ?? 1_000;
	let state: SessionStoreState = {
		session: options.initialSession ?? null,
		connection: 'idle',
		sending: false,
		error: null
	};
	const store = writable(state);
	let controller: AbortController | null = null;
	let startPromise: Promise<void> | null = null;

	function update(patch: Partial<SessionStoreState>): void {
		state = { ...state, ...patch };
		store.set(state);
	}

	function endpoint(path: string): string {
		return `${baseUrl}/${path}`;
	}

	async function requestSnapshot(path: string, init?: RequestInit): Promise<SessionSnapshot> {
		const headers = new Headers(init?.headers);
		headers.set('Accept', 'application/json');
		headers.set('Authorization', `Bearer ${options.key}`);
		const response = await fetcher(endpoint(path), {
			...init,
			headers
		});

		if (!response.ok) throw await responseError(response);
		const snapshot: unknown = await response.json();
		if (!isSessionSnapshot(snapshot)) throw new Error('Invalid session response');
		return snapshot;
	}

	async function hydrate(signal?: AbortSignal): Promise<SessionSnapshot> {
		try {
			const session = await requestSnapshot('session', { signal });
			update({ session, error: null });
			return session;
		} catch (error) {
			if (!signal?.aborted) update({ error: errorMessage(error) });
			throw error;
		}
	}

	async function stream(signal: AbortSignal): Promise<void> {
		let connected = false;

		while (!signal.aborted) {
			update({ connection: connected ? 'reconnecting' : 'connecting' });

			try {
				const response = await fetcher(endpoint('session/events'), {
					headers: {
						Accept: 'text/event-stream',
						Authorization: `Bearer ${options.key}`
					},
					signal
				});

				if (!response.ok) throw await responseError(response);
				const contentType = response.headers.get('content-type');
				if (!contentType?.includes('text/event-stream')) {
					throw new Error('The session event endpoint did not return an event stream');
				}

				connected = true;
				update({ connection: 'open', error: null });
				await consumeEventStream(
					response,
					(event) => {
						if (event.event !== 'session') return;
						const session: unknown = JSON.parse(event.data);
						if (isSessionSnapshot(session)) update({ session, error: null });
					},
					signal
				);

				if (!signal.aborted) throw new Error('The session event stream closed');
			} catch (error) {
				if (signal.aborted) break;
				update({ connection: 'reconnecting', error: errorMessage(error) });
				await delay(reconnectDelayMs, signal);
			}
		}

		update({ connection: 'closed' });
	}

	async function start(): Promise<void> {
		if (startPromise) return startPromise;
		if (controller) return;

		controller = new AbortController();
		const signal = controller.signal;
		update({ connection: 'connecting', error: null });

		startPromise = (async () => {
			try {
				if (!state.session) await hydrate(signal);
				if (!signal.aborted) void stream(signal);
			} catch (error) {
				controller = null;
				update({ connection: 'closed' });
				throw error;
			}
		})();

		try {
			await startPromise;
		} finally {
			startPromise = null;
		}
	}

	async function sendMessage(content: string): Promise<SessionSnapshot> {
		const message = content.trim();
		if (!message) throw new Error('Message cannot be empty');

		update({ sending: true, error: null });
		try {
			const session = await requestSnapshot('session/messages', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ content: message })
			});
			update({ session });
			return session;
		} catch (error) {
			update({ error: errorMessage(error) });
			throw error;
		} finally {
			update({ sending: false });
		}
	}

	function stop(): void {
		controller?.abort();
		controller = null;
		update({ connection: 'closed' });
	}

	return {
		subscribe: store.subscribe,
		hydrate: () => hydrate(),
		start,
		sendMessage,
		stop
	};
}
