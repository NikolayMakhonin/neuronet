import {cycle} from "./helpers";

export function learn<TInput, TOutput>({
	nextInput,
	func,
	expectedFunc,
	fixError,
	maxIterations,
	maxTime,
}: {
	nextInput: (prevInput: TInput, iteration: number) => TInput,
	func: (input: TInput) => TOutput,
	expectedFunc: (input: TInput) => TOutput,
	fixError: (input: TInput, output: TOutput, expectedOutput: TOutput) => boolean|void,
	maxIterations?: number,
	maxTime?: number,
}) {
	let input: TInput

	cycle({
		maxIterations,
		maxTime,
		func(iteration) {
			input = nextInput(input, iteration)
			const output = func(input)
			const expectedOutput = expectedFunc(input)
			return fixError(input, output, expectedOutput)
		}
	})
}
