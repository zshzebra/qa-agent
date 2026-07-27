<script lang="ts">
	import { createSessionClient } from '$lib';
	import Button from '$lib/components/ui/button/button.svelte';
	import Input from '$lib/components/ui/input/input.svelte';
	import Skeleton from '$lib/components/ui/skeleton/skeleton.svelte';
	import DOMPurify from 'dompurify';
	import { marked } from 'marked';

	const agent = createSessionClient({
		key: 'client-a'
	});

	let message = $state('');

	$effect(() => {
		void agent.start();
		return () => agent.stop();
	});

	async function handleSubmit(event: SubmitEvent) {
		event.preventDefault();
		await agent.sendMessage(message.trim());
		message = '';
	}
</script>

<div class="max-w-prose p-6 h-screen flex flex-col mx-auto">
	<div class="h-full overflow-y-scroll">
		<div class="flex flex-col gap-4">
			{#if $agent.session}
				{#each $agent.session.messages as item, index (index)}
					<div class={item.role == 'user' ? 'p-2 bg-accent text-right w-max ml-auto' : null}>
						<!-- eslint-disable-next-line svelte/no-at-html-tags -->
						{@html DOMPurify.sanitize(marked.parse(item.content) as string)}
					</div>
				{/each}
				{#if $agent.session.status == 'working'}
					<div class="flex flex-col gap-2">
						<!-- NOTE: Slightly overkill way to have the two skeleton loaders -->
						<!-- different widths on _practically_ all devices -->
						<Skeleton class="h-4 max-w-48 w-full" />
						<div class="flex flex-row flex-nowrap">
							<Skeleton class="h-4 max-w-40 w-full" />
							<div class="w-8"></div>
						</div>
					</div>
				{/if}
			{/if}
		</div>
	</div>

	<form onsubmit={handleSubmit} class="flex flex-row">
		<Input bind:value={message} placeholder="Amazo..." />
		<Button disabled={$agent.sending}>Send</Button>
	</form>
</div>
