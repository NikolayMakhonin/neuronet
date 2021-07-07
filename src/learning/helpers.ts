export function cycle({
	func,
	maxIterations,
	maxTime,
}: {
	func: (iteration: number, elapsedTime: number, last: boolean) => boolean|void,
	maxIterations?: number,
	maxTime?: number,
}) {
	let timer
	try {
		let startTime: number
		let endTime: number
		let now: number
		now = Date.now()
		startTime = now
		endTime = maxTime ? startTime + maxTime : 0
		timer = setInterval(() => {
			now = Date.now()
		}, 100)

		let iteration = 0
		let end: boolean = false
		while (!end) {
			end ||= endTime && now >= endTime || maxIterations && iteration >= maxIterations - 1
			end = !!func(iteration, now - startTime, end) || end
			iteration++
		}
	} finally {
		if (timer) {
			clearInterval(timer)
		}
	}
}
