export type SessionStatus = 'working' | 'waiting_for_user' | 'completed' | 'failed';

export type SessionConnection = 'idle' | 'connecting' | 'open' | 'reconnecting' | 'closed';

export interface SessionMessage {
	role: 'user' | 'assistant';
	content: string;
}

/** A researcher that is running now. */
export interface ActiveResearcher {
	id: string;
	/** The domains of every source used by a researcher */
	domains: string[];
}

export interface SessionSnapshot {
	revision: number;
	status: SessionStatus;
	messages: SessionMessage[];
	completion_message: string | null;
	result: Record<string, unknown> | null;
	error: string | null;
	/** The active researchers in launch order */
	researchers: ActiveResearcher[];
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
