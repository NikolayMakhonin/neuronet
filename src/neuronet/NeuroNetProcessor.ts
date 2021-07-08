import {learn} from "../learning/learn";
import {TNeuroNet} from "./contracts";

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
		maxIterations,
		maxTime,
	}: {
		nextInputValue: (inputIndex: number, inputCount: number, iteration: number) => number|null,
		learningRate: number,
		maxIterations?: number,
		maxTime?: number,
	}) {
		_learnNeuroNet({
			neuroNet: this.neuroNet,
			nextInputValue,
			expectedFunc: this.expectedFunc,
			learningRate,
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
	maxIterations,
	maxTime,
}: {
	neuroNet: TNeuroNet<TInput>,
	nextInputValue: (inputIndex: number, inputCount: number, iteration: number) => number|null,
	expectedFunc: (input: TInput, output: TOutput) => void,
	learningRate?: number,
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

			let error = 0 // результат функции ошибок
			for (let i = 0, len = output.length; i < len; i++) {
				const actual = output[i]
				const expected = expectedOutput[i]
				error += (actual - expected) ** 2
			}

			errorSumSqr += error
			errorCount++

			if (learningRate) {
				for (let i = 0, len = output.length; i < len; i++) {
					const neuron = lastLayer[i]
					const yi = output[i]
					const ai = expectedOutput[i]
					const dDk_dyj = 2 * (yi - ai) // частная производная функции ошибок
					neuron.clear_dE_Dw()
					const sum_sqr_dE_Dw = neuron.calc_dE_Dw(dDk_dyj)
					if (sum_sqr_dE_Dw !== 0) {
						neuron.changeWeights(learningRate * Math.sqrt(error / sum_sqr_dE_Dw))
					}
				}
			}
		},
	})

	return errorSumSqr
}
