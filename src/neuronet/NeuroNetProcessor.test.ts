import assert from "assert";
import {Neuron} from "../neuron/neuron";
import {funcs} from "../neuron/funcs";
import {NeuroNetProcessor} from "./NeuroNetProcessor";

describe('NeuroNetProcessor', function() {
	it('single neuron', function() {
		const input = [0]
		const neuron = new Neuron(funcs.sigmoid, input, [0])
		const neuroNet = new NeuroNetProcessor({
			neuroNet: {
				input,
				layers: [[neuron]],
			},
			expectedFunc(input, output) {
				output[0] = funcs.sigmoid.func(input[0])
			},
		})

		neuroNet.learn({
			nextInputValue(inputIndex, inputCount) {
				return Math.random() * 5 - 10
			},
			learningRate: 0.1,
			maxIterations: 100000,
		})

		const error = neuroNet.calcError({
			nextInputValue(inputIndex, inputCount) {
				return (inputIndex / inputCount) * 10 - 5
			},
			maxIterations: 1000,
		})

		console.log('error=' + error)
	})
})
