import {TNeuronFunc} from "../neuron/contracts";
import {Neuron} from "../neuron/neuron";
import {TNeuroNet, TNeuroNetInput} from "./contracts";

export function createPerceptron<TInput extends TNeuroNetInput = TNeuroNetInput>({
	input,
	countLayers,
	getLayerSize,
	getLinkWeight,
	getNeuronFunc,
}: {
	input: TInput,
	countLayers: number,
	getLayerSize: (countLayers: number, layerIndex: number) => number,
	getLinkWeight: (countLayers: number, layerIndex: number, countNeurons: number, neuronIndex: number, countLinks: number, linkIndex: number) => number,
	getNeuronFunc: (countLayers: number, layerIndex: number, countNeurons: number, neuronIndex: number) => TNeuronFunc,
}): TNeuroNet<TInput> {
	const layers: Neuron[][] = []
	let prevLayer: (Neuron|number)[] = input
	for (let layerIndex = 0; layerIndex < countLayers; layerIndex++) {
		const countNeurons = getLayerSize(countLayers, layerIndex)
		const layer: Neuron[] = []
		for (let neuronIndex = 0; neuronIndex < countNeurons; neuronIndex++) {
			const neuronFunc = getNeuronFunc(countLayers, layerIndex, countNeurons, neuronIndex)
			const weights = []
			for (let linkIndex = 0, countLinks = prevLayer.length; linkIndex < countLinks; linkIndex++) {
				weights[linkIndex] = getLinkWeight(countLayers, layerIndex, countNeurons, neuronIndex, countLinks, linkIndex)
			}
			const neuron = new Neuron(neuronFunc, prevLayer, weights)
			layer.push(neuron)
		}
		prevLayer = layer
		layers.push(layer)
	}
	return {
		input,
		layers,
	}
}
