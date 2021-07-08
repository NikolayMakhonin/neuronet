import assert from "assert";
import {Neuron} from "../neuron/Neuron";
import {funcs} from "../neuron/funcs";
import {NeuroNetProcessor} from "./NeuroNetProcessor";
import {TNeuronFunc} from "../neuron/contracts";

describe('NeuroNetProcessor', function() {
	function testSingleNeuron({
		name,
		learningRate,
		maxIterations,
		func,
		minError,
		maxError,
		testsCount,
	}: {
		name: string,
		learningRate: number,
		maxIterations: number,
		func: TNeuronFunc,
		minError: number,
		maxError: number,
		testsCount: number,
	}) {
		let testMaxError = 0

		for (let i = 0; i < testsCount; i++) {
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
			if (error > testMaxError) {
				testMaxError = error
			}

			assert.ok(error <= maxError, `Single neuron (${name}); error=${error}`)
		}

		assert.ok(testMaxError >= minError, `Single neuron (${name}); error=${testMaxError}`)

		console.log(`Single neuron (${name}); error=${testMaxError}`)
	}

	it('single neuron sigmoid', function() {
		testSingleNeuron({
			name: 'sigmoid',
			learningRate: 0.5,
			maxIterations: 100,
			func: funcs.sigmoid,
			minError: 0.000001,
			maxError: 0.0001,
			testsCount: 1000,
		})
	})

	it('single neuron ReLU', function() {
		testSingleNeuron({
			name: 'ReLU',
			learningRate: 0.5,
			maxIterations: 100,
			func: funcs.ReLU,
			minError: 1e-20,
			maxError: 0.0001,
			testsCount: 1000,
		})
	})

	xit('single neuron sin', function() {
		testSingleNeuron({
			name: 'sin',
			learningRate: 0.1,
			maxIterations: 2000,
			func: funcs.sin,
			minError: 0,
			maxError: 0.0001,
			testsCount: 10000,
		})
	})
})
