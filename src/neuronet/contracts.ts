import {Neuron} from "../neuron/Neuron";
import {TNeuronInput} from "../neuron/contracts";

export type TNeuroNet<TInput extends (Neuron|number)[] = (Neuron|number)[]> = {
	input: TInput
	layers: Neuron[][]
}
export type TNeuroNetInput = TNeuronInput
