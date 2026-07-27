export interface ServerSentEvent {
	id?: string;
	event?: string;
	data: string;
}

function parseEvent(block: string): ServerSentEvent | null {
	let id: string | undefined;
	let event: string | undefined;
	const data: string[] = [];

	for (const line of block.split(/\r?\n/)) {
		if (!line || line.startsWith(':')) continue;

		const separator = line.indexOf(':');
		const field = separator === -1 ? line : line.slice(0, separator);
		let value = separator === -1 ? '' : line.slice(separator + 1);
		if (value.startsWith(' ')) value = value.slice(1);

		if (field === 'id') id = value;
		else if (field === 'event') event = value;
		else if (field === 'data') data.push(value);
	}

	if (data.length === 0) return null;
	return { id, event, data: data.join('\n') };
}

export async function consumeEventStream(
	response: Response,
	onEvent: (event: ServerSentEvent) => void,
	signal: AbortSignal
): Promise<void> {
	if (!response.body) throw new Error('The event stream has no response body');

	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let buffer = '';

	try {
		while (!signal.aborted) {
			const { value, done } = await reader.read();
			if (done) break;

			buffer += decoder.decode(value, { stream: true });
			const blocks = buffer.split(/\r?\n\r?\n/);
			buffer = blocks.pop() ?? '';

			for (const block of blocks) {
				const event = parseEvent(block);
				if (event) onEvent(event);
			}
		}
	} finally {
		reader.releaseLock();
	}
}
