<script lang="ts">
	import type { ActiveResearcher } from '$lib';
	import { fade, scale } from 'svelte/transition';
	import { monogram } from './monogram';

	let { researchers, max = 3 }: { researchers: ActiveResearcher[]; max?: number } = $props();
</script>

<div
	class="flex min-h-13 flex-row flex-nowrap items-center gap-2 overflow-x-auto py-2"
	aria-hidden="true"
>
	{#each researchers as researcher (researcher.id)}
		{@const start = Math.max(researcher.domains.length - max, 0)}
		<div class="flex shrink-0 items-center gap-1 rounded-full bg-muted p-2" transition:fade>
			<svg
				class="size-4 animate-pulse text-muted-foreground"
				viewBox="0 0 24 24"
				fill="none"
				stroke="currentColor"
				stroke-width="2"
				aria-hidden="true"
			>
				<circle cx="11" cy="11" r="7" />
				<path d="m20 20-4-4" />
			</svg>
			{#each researcher.domains.slice(start) as domain, i (start + i)}
				{@const icon = monogram(domain)}
				<span
					class="flex size-5 items-center justify-center rounded-full text-[0.625rem] font-semibold text-white"
					style:background-color="hsl({icon.hue} 55% 45%)"
					title={domain}
					in:scale={{ delay: i * 80 }}
				>
					{icon.letter}
				</span>
			{/each}
			{#if start > 0}
				<span class="text-xs text-muted-foreground">+{start}</span>
			{/if}
		</div>
	{/each}
</div>
