import {Neuron} from "./neuron";

export type TNeuronFunc = {
	func: (input: number) => number
	derivative: (input: number) => number
}

export type TNeuronInput = (Neuron|number)[]
