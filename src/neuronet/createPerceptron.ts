import {TNeuronFunc} from "../neuron/contracts";
import {Neuron} from "../neuron/Neuron";
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
	getLinkWeight: (countLayers: number, layerIndex: number, countNeurons: number, neuronIndex: number, countLinks: number, linkIndex: number) => {
		value: number,
		fixed?: boolean,
	},
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
			const fixedWeights = []
			for (let linkIndex = 0, countLinks = prevLayer.length; linkIndex < countLinks; linkIndex++) {
				const {value, fixed} = getLinkWeight(countLayers, layerIndex, countNeurons, neuronIndex, countLinks, linkIndex)
				weights[linkIndex] = value
				fixedWeights[linkIndex] = fixed
			}
			const neuron = new Neuron(neuronFunc, prevLayer, weights, fixedWeights)
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
