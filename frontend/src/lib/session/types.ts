export type SessionStatus = 'working' | 'waiting_for_user' | 'completed' | 'failed';

export type SessionConnection = 'idle' | 'connecting' | 'open' | 'reconnecting' | 'closed';

export interface SessionMessage {
	role: 'user' | 'assistant';
	content: string;
}

export interface SessionSnapshot {
	revision: number;
	status: SessionStatus;
	messages: SessionMessage[];
	completion_message: string | null;
	result: Record<string, unknown> | null;
	error: string | null;
}

export interface SessionStoreState {
	session: SessionSnapshot | null;
	connection: SessionConnection;
	sending: boolean;
	error: string | null;
}

export interface SessionClientOptions {
	key: string;
	baseUrl?: string;
	initialSession?: SessionSnapshot;
	fetch?: typeof globalThis.fetch;
	reconnectDelayMs?: number;
}
