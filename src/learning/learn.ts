export function learn<TInput, TOutput, TError>({

}: {
	func: (input: TInput) => TOutput,
	expectedFunc: (input: TInput) => TOutput,
	calcError: (output: TOutput, expectedOutput: TOutput) => TError,
	fixError: (output: TOutput, expectedOutput: TOutput, error: TError) => void,
	maxIterations?: number,
	maxTime?: number,
}) {

}
