/** Use a letter and hue to represent a website */
export interface Monogram {
	letter: string;
	hue: number;
}

export function monogram(domain: string): Monogram {
	let hash = 0;
	for (const char of domain) hash = (hash * 31 + char.codePointAt(0)!) >>> 0;
	return { letter: (domain[0] ?? '?').toUpperCase(), hue: hash % 360 };
}
