const infinity = 1e10

export function fixInfinity(value: number) {
	if (value >= infinity) {
		return 1e10
	}
	if (value <= -infinity) {
		return -1e10
	}
	if (Number.isNaN(value)) {
		throw new Error('value is NaN')
	}
	return value
}
