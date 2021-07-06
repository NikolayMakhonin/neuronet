import {INeuronFunc} from "../neuron/contracts";

export function createPerceptron({

}: {
	countLayers: number,
	getLayerSize: (layerIndex: number) => number,
	getNeuronFunc: (layerIndex: number, neuronIndex: number) => INeuronFunc,
}) {

}
