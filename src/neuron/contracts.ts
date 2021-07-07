import {Neuron} from "./neuron";

export interface INeuronFunc {
	func: (input: number) => number
	derivative: (input: number) => number
}

export type TNeuronInput = (Neuron|number)[]
