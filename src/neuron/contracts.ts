export interface INeuronFunc {
	func: (input: number) => number
	derivative: (input: number) => number
}
