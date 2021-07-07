import {learn} from "../learning/learn";
import {TNeuroNet} from "./contracts";

type TInput = number[]
type TOutput = number[]

export function learnNeuroNet({
	neuroNet,
	nextInputValue,
	expectedFunc,
	learningRate,
	maxIterations,
	maxTime,
}: {
	neuroNet: TNeuroNet<TInput>,
	nextInputValue: (inputIndex: number, inputCount: number) => number,
	expectedFunc: (input: TInput) => TOutput,
	learningRate: number,
	maxIterations?: number,
	maxTime?: number,
}) {
	const lastLayer = neuroNet.layers[neuroNet.layers.length - 1]
	const output: TOutput = []

	learn<TInput, TOutput>({
		maxIterations,
		maxTime,
		nextInput(input) {
			if (!input) {
				return neuroNet[0]
			}
			for (let i = 0, len = input.length; i < len; i++) {
				input[i] = nextInputValue(i, len)
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
		expectedFunc,
		fixError(input, output, expectedOutput) {
			for (let i = 0, len = output.length; i < len; i++) {
				const neuron = lastLayer[i]

				const actual = output[i]
				const expected = expectedOutput[i]
				const error = (actual - expected) ** 2

				const dE_do_j = 2 * (actual - expected) * learningRate
				neuron.clear_dE_Dw()
				const sum_sqr_dE_Dw = neuron.calc_dE_Dw(dE_do_j)

				neuron.changeWeights(learningRate)
			}
		},
	})
}
