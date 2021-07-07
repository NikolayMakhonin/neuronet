import assert from "assert";
import {Neuron} from "../neuron/neuron";
import {funcs} from "../neuron/funcs";
import {NeuroNetProcessor} from "./NeuroNetProcessor";
import {TNeuronFunc} from "../neuron/contracts";

describe('NeuroNetProcessor', function() {
	function testSingleNeuron({
		name,
		learningRate,
		maxIterations,
		func,
		maxError,
	}: {
		name: string,
		learningRate: number,
		maxIterations: number,
		func: TNeuronFunc,
		maxError: number,
	}) {
		const input = [0]
		const neuron = new Neuron(func, input, [0])
		const neuroNet = new NeuroNetProcessor({
			neuroNet: {
				input,
				layers: [[neuron]],
			},
			expectedFunc(input, output) {
				output[0] = func.func(input[0])
			},
		})

		neuroNet.learn({
			nextInputValue(inputIndex, inputCount) {
				return Math.random() * 10 - 5
			},
			learningRate,
			maxIterations,
		})

		const calcErrorIterations = 1000
		const error = neuroNet.calcError({
			nextInputValue(inputIndex, inputCount, iteration) {
				return (iteration / calcErrorIterations) * 10 - 5
			},
			maxIterations: calcErrorIterations,
		})

		console.log(`Single neuron (${name}); error=${error}`)

		assert.ok(error < maxError)
	}

	it('single neuron sigmoid', function() {
		testSingleNeuron({
			name: 'sigmoid',
			learningRate: 0.1,
			maxIterations: 150,
			func: funcs.sigmoid,
			maxError: 0.0001,
		})
	})

	it('single neuron ReLU', function() {
		testSingleNeuron({
			name: 'ReLU',
			learningRate: 0.1,
			maxIterations: 400,
			func: funcs.ReLU,
			maxError: 0.0001,
		})
	})
})
