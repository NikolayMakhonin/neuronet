import {learn} from "../learning/learn";
import {TNeuroNet} from "./contracts";
import {Neuron} from '../neuron/Neuron'

type TInput = number[]
type TOutput = number[]

export class NeuroNetProcessor {
	readonly neuroNet: TNeuroNet<TInput>
	readonly expectedFunc: (input: TInput, output: TOutput) => void

	constructor({
		neuroNet,
		expectedFunc,
	}: {
		neuroNet: TNeuroNet<TInput>,
		expectedFunc: (input: TInput, output: TOutput) => void,
	}) {
		this.neuroNet = neuroNet
		this.expectedFunc = expectedFunc
	}

	learn({
		nextInputValue,
		learningRate,
		momentRate,
		maxIterations,
		maxTime,
	}: {
		nextInputValue: (inputIndex: number, inputCount: number, iteration: number) => number|null,
		learningRate: number,
		momentRate?: number,
		maxIterations?: number,
		maxTime?: number,
	}) {
		_learnNeuroNet({
			neuroNet: this.neuroNet,
			nextInputValue,
			expectedFunc: this.expectedFunc,
			learningRate,
			momentRate,
			maxIterations,
			maxTime,
		})
	}

	calcError({
		nextInputValue,
		maxIterations,
		maxTime,
	}: {
		nextInputValue: (inputIndex: number, inputCount: number, iteration: number) => number|null,
		maxIterations?: number,
		maxTime?: number,
	}) {
		return _learnNeuroNet({
			neuroNet: this.neuroNet,
			nextInputValue,
			expectedFunc: this.expectedFunc,
			maxIterations,
			maxTime,
		})
	}
}

function _learnNeuroNet({
	neuroNet,
	nextInputValue,
	expectedFunc,
	learningRate,
	momentRate,
	maxIterations,
	maxTime,
}: {
	neuroNet: TNeuroNet<TInput>,
	nextInputValue: (inputIndex: number, inputCount: number, iteration: number) => number|null,
	expectedFunc: (input: TInput, output: TOutput) => void,
	learningRate?: number,
	momentRate?: number,
	maxIterations?: number,
	maxTime?: number,
}): number {
	const lastLayer = neuroNet.layers[neuroNet.layers.length - 1]
	const output: TOutput = []
	const expectedOutput: TOutput = []
	let errorSumSqr = 0
	let errorCount = 0

	learn<TInput, TOutput>({
		maxIterations,
		maxTime,
		nextInput(input, iteration) {
			if (!input) {
				input = neuroNet.input
			}
			for (let i = 0, len = input.length; i < len; i++) {
				const newValue = nextInputValue(i, len, iteration)
				if (newValue != null) {
					input[i] = newValue
				}
			}
			return input
		},
		func() {
			for (let i = 0, len = lastLayer.length; i < len; i++) {
				const neuron = lastLayer[i]
				neuron.calcOutput()
				output[i] = neuron.output
			}
			return output
		},
		expectedFunc(input) {
			expectedFunc(input, expectedOutput)
			return expectedOutput
		},
		fixError(input, output, expectedOutput) {
			// see: https://www.youtube.com/watch?v=mG8A-k9cDiU
			// see: https://www.youtube.com/watch?v=87gux0d36bw
			// see: https://www.youtube.com/watch?v=W2LshGngCNw

			const error = calcError(output, expectedOutput)

			errorSumSqr += error
			errorCount++

			if (learningRate) {
				clear_dE_Dw(neuroNet.layers)
				for (let i = 0, len = lastLayer.length; i < len; i++) {
					const neuron = lastLayer[i]
					const yi = neuron.output
					const ai = expectedOutput[i]
					const dDk_dyj = 2 * (yi - ai) // часть производной функции ошибок
					neuron.calc_dE_Dw(dDk_dyj)
				}

				const sum_sqr_dE_Dw = calcSumSqr_dE_Dw(neuroNet.layers)
				if (sum_sqr_dE_Dw !== 0) {
					const dw = learningRate * Math.sqrt(error / sum_sqr_dE_Dw)
					changeWeights(neuroNet.layers, dw, momentRate)
				}
			}
		},
	})

	return errorSumSqr
}

function perceptronForEach(
	layers: Neuron[][],
	func: (neuron: Neuron, layerIndex: number, neuronIndex: number, layers: Neuron[][]) => void,
) {
	for (let layerIndex = 0, layersLength = layers.length; layerIndex < layersLength; layerIndex++) {
		const neurons = layers[layerIndex]
		for (let neuronIndex = 0, neuronLength = neurons.length; neuronIndex < neuronLength; neuronIndex++) {
			const neuron = neurons[neuronIndex]
			func(neuron, layerIndex, neuronIndex, layers)
		}
	}
}

// функция ошибок (сумма квадратов разности)
function calcError(actualOutput: (Neuron|number)[], expectedOutput: number[]) {
	let error = 0
	for (let i = 0, len = actualOutput.length; i < len; i++) {
		let actual = actualOutput[i]
		if (typeof actual !== 'number') {
			actual = actual.output
		}
		const expected = expectedOutput[i]
		error += (actual - expected) ** 2
	}
	return error
}

function clear_dE_Dw(layers: Neuron[][]) {
	perceptronForEach(layers, neuron => {
		neuron.clear_dE_Dw()
	})
}

function calcSumSqr_dE_Dw(layers: Neuron[][]) {
	let sum_sqr_dE_Dw = 0
	perceptronForEach(layers, neuron => {
		sum_sqr_dE_Dw += neuron.calcSumSqr_dE_Dw()
	})
	return sum_sqr_dE_Dw
}

function changeWeights(layers: Neuron[][], dw: number, momentRate?: number) {
	perceptronForEach(layers, neuron => {
		neuron.changeWeights(dw, momentRate)
	})
}
