import assert from "assert";
import {createPerceptron} from "./createPerceptron";
import {funcs} from "../neuron/funcs";

describe('createPerceptron', function() {
	it('base', function() {
		const _funcs = [funcs.sin, funcs.sigmoid, funcs.exp]
		const neuronet = createPerceptron({
			input: [1, 2],
			countLayers: 3,
			getLayerSize(countLayers, layerIndex) {
				return countLayers - layerIndex
			},
			getLinkWeight(countLayers, layerIndex, countNeurons, neuronIndex, countLinks, linkIndex) {
				return { value: layerIndex + neuronIndex + linkIndex + 1 }
			},
			getNeuronFunc(countLayers, layerIndex, countNeurons, neuronIndex) {
				return _funcs[neuronIndex]
			},
		})

		const lastLayer = neuronet.layers[neuronet.layers.length - 1]
		assert.strictEqual(lastLayer.length, 1)
		lastLayer[0].calcOutput()

		const l0 = [
			1,
			2,
		]

		const l1 = [
			funcs.sin.func(l0[0] * 1 + l0[1] * 2),
			funcs.sigmoid.func(l0[0] * 2 + l0[1] * 3),
			funcs.exp.func(l0[0] * 3 + l0[1] * 4),
		]

		const l2 = [
			funcs.sin.func(l1[0] * 2 + l1[1] * 3 + l1[2] * 4),
			funcs.sigmoid.func(l1[0] * 3 + l1[1] * 4 + l1[2] * 5),
		]

		const l3 = [
			funcs.sin.func(l2[0] * 3 + l2[1] * 4),
		]

		assert.strictEqual(lastLayer[0].output, l3[0])
	})
})
