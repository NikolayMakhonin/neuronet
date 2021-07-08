import assert from "assert";
import {Neuron} from "../neuron/Neuron";
import {funcs} from "../neuron/funcs";
import {NeuroNetProcessor} from "./NeuroNetProcessor";
import {TNeuronFunc} from "../neuron/contracts";
import {createPerceptron} from "./createPerceptron";

describe('NeuroNetProcessor', function() {
	this.timeout(60 * 60 * 1000)

	function test({
		name,
		neuroNet,
		learningRate,
		maxIterations,
		valueRange,
		inputIsFixed,
		minError,
		maxError,
		testsCount,
	}: {
		name: string,
		neuroNet: NeuroNetProcessor,
		learningRate: number,
		maxIterations: number,
		valueRange: [number, number],
		inputIsFixed?: (inputIndex, inputCount) => boolean,
		minError: number,
		maxError: number,
		testsCount: number,
	}) {
		let testMaxError = 0

		for (let i = 0; i < testsCount; i++) {

			neuroNet.learn({
				nextInputValue(inputIndex, inputCount) {
					if (inputIsFixed && inputIsFixed(inputIndex, inputCount)) {
						return null
					}
					return Math.random() * (valueRange[1] - valueRange[0]) + valueRange[0]
				},
				learningRate,
				maxIterations,
			})

			const calcErrorIterations = 1000
			const error = neuroNet.calcError({
				nextInputValue(inputIndex, inputCount, iteration) {
					if (inputIsFixed && inputIsFixed(inputIndex, inputCount)) {
						return null
					}
					return (iteration / calcErrorIterations) * (valueRange[1] - valueRange[0]) + valueRange[0]
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

	function createSingleNeuron(func: TNeuronFunc) {
		const input = [0]
		const neuron = new Neuron(func, input, [0])
		return new NeuroNetProcessor({
			neuroNet: {
				input,
				layers: [[neuron]],
			},
			expectedFunc(input, output) {
				output[0] = func.func(input[0])
			},
		})
	}

	describe('single neuron', function() {
		it('sigmoid', function () {
			test({
				name: 'sigmoid',
				neuroNet: createSingleNeuron(funcs.sigmoid),
				learningRate: 0.5,
				maxIterations: 100,
				valueRange: [-5, 5],
				minError: 0.000001,
				maxError: 0.0001,
				testsCount: 1000,
			})
		})

		it('single neuron ReLU', function () {
			test({
				name: 'ReLU',
				neuroNet: createSingleNeuron(funcs.ReLU),
				learningRate: 0.5,
				maxIterations: 100,
				valueRange: [-5, 5],
				minError: 0,
				maxError: 0.0001,
				testsCount: 1000,
			})
		})
	})

	describe('fourier', function() {
		function createFourier({
			freqs,
			startWeights,
			expectedFunc,
			considerPhase,
		}: {
			freqs: number[],
			startWeights: number[],
			expectedFunc: (x: number) => number,
			considerPhase?: boolean,
		}) {
			const input = [0, 1]
			return new NeuroNetProcessor({
				neuroNet: createPerceptron({
					input,
					countLayers: 2,
					getLayerSize(countLayers, layerIndex) {
						if (layerIndex > 0) {
							return 1
						} else {
							return freqs.length * (considerPhase ? 2 : 1)
						}
					},
					getLinkWeight(countLayers, layerIndex, countNeurons, neuronIndex, countLinks, linkIndex) {
						if (layerIndex > 0) {
							return {
								value: startWeights[linkIndex], //linkIndex === 0 ? 0 : 1,
							}
						} else if (considerPhase) {
							if (linkIndex > 0) {
								return {
									value: neuronIndex % 2 ? Math.PI / 2 : 0,
									fixed: true,
								}
							} else {
								return {value: freqs[(neuronIndex / 2) | 0], fixed: true}
							}
						} else {
							if (linkIndex > 0) {
								return {
									value: 0,
									fixed: true,
								}
							} else {
								return {value: freqs[neuronIndex], fixed: true}
							}
						}
					},
					getNeuronFunc(countLayers, layerIndex, countNeurons, neuronIndex) {
						return layerIndex > 0
							? funcs.x
							: funcs.sin
					},
				}),
				expectedFunc(input, output) {
					output[0] = expectedFunc(input[0])
				},
			})
		}

		it('freqs[1]', function () {
			test({
				name: 'fourier[1]',
				neuroNet: createFourier({
					freqs: [1],
					startWeights: [0],
					expectedFunc: o => Math.sin(o),
				}),
				inputIsFixed: (index) => index === 1,
				learningRate: 0.1,
				maxIterations: 200,
				valueRange: [-5, 5],
				minError: 0,
				maxError: 0.0001,
				testsCount: 1000,
			})
		})

		it('freqs[1] with phase', function () {
			test({
				name: 'fourier[1] with phase',
				neuroNet: createFourier({
					freqs: [1],
					startWeights: [0, 0],
					expectedFunc: o => Math.sin(o) + Math.sin(o + Math.PI / 2),
					considerPhase: true,
				}),
				inputIsFixed: (index) => index === 1,
				learningRate: 0.1,
				maxIterations: 200,
				valueRange: [-5, 5],
				minError: 0,
				maxError: 0.0001,
				testsCount: 1000,
			})
		})

		it('freqs[1,2] with phase', function () {
			test({
				name: 'fourier[1,2] with phase',
				neuroNet: createFourier({
					freqs: [1, 2],
					startWeights: [0, 0, 0, 0],
					expectedFunc: o =>
						Math.sin(o) + Math.sin(o + Math.PI / 2) +
						Math.sin(2 * o) + Math.sin(2 * o + Math.PI / 2),
					considerPhase: true,
				}),
				inputIsFixed: (index) => index === 1,
				learningRate: 0.1,
				maxIterations: 350,
				valueRange: [-5, 5],
				minError: 0,
				maxError: 0.0001,
				testsCount: 1000,
			})
		})

		it('freqs[1,2,3,4] with phase', function () {
			test({
				name: 'fourier[1,2,3,4] with phase',
				neuroNet: createFourier({
					freqs: [1, 2, 3, 4],
					startWeights: [0, 0, 0, 0, 0, 0, 0, 0],
					expectedFunc: o =>
						Math.sin(o) + Math.sin(o + Math.PI / 2) +
						Math.sin(2 * o) + Math.sin(2 * o + Math.PI / 2) +
						Math.sin(3 * o) + Math.sin(3 * o + Math.PI / 2) +
						Math.sin(4 * o) + Math.sin(4 * o + Math.PI / 2),
					considerPhase: true,
				}),
				inputIsFixed: (index) => index === 1,
				learningRate: 0.1,
				maxIterations: 550,
				valueRange: [-5, 5],
				minError: 0,
				maxError: 0.0001,
				testsCount: 1000,
			})
		})
	})
})
