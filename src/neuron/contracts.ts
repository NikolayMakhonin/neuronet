import {Neuron} from "./Neuron";

export type TNeuronFunc = {
	func: (input: number) => number
	derivative: (input: number) => number
}

export type TNeuronInput = (Neuron|number)[]
