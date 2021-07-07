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
		nextInputValue: (inputIndex: number, inputCount: number, iteration: number) => number,
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
		nextInputValue: (inputIndex: number, inputCount: number, iteration: number) => number,
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
	nextInputValue: (inputIndex: number, inputCount: number, iteration: number) => number,
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
				input[i] = nextInputValue(i, len, iteration)
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
			for (let i = 0, len = output.length; i < len; i++) {
				const neuron = lastLayer[i]

				const actual = output[i]
				const expected = expectedOutput[i]
				const error = (actual - expected) ** 2

				errorSumSqr += error
				errorCount++

				if (learningRate) {
					const dE_do_j = 2 * (actual - expected) * 1
					neuron.clear_dE_Dw()
					const sum_sqr_dE_Dw = neuron.calc_dE_Dw(dE_do_j)
					if (sum_sqr_dE_Dw !== 0) {
						neuron.changeWeights(learningRate * error / sum_sqr_dE_Dw)
					}
				}
			}
		},
	})

	return errorSumSqr
}
