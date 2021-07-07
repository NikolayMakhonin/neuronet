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

				const dE_do_j = 2 * (actual - expected)
				neuron.clear_dE_Dw()
				const sum_sqr_dE_Dw = neuron.calc_dE_Dw(dE_do_j)

				if (sum_sqr_dE_Dw !== 0) {
					neuron.changeWeights(learningRate * error / sum_sqr_dE_Dw)
				}
			}
		},
	})

	// const minInputValue = -2 // 1e-10
	// const maxInputValue = 4 // 10
	// const learningRate = 1
	// const countIterations = 100000000
	// const minError = 1e-13
	// let time = Date.now()
	// let prev_sum_sqr_dE_Dw
	// let prev_linearError
	// let iteration_sum_sqr_dE_Dw = 0
	// let iteration_error = 0
	//
	// for (let n = 0; n < countIterations; n++) {
	// 	fillInputsRandom(inputs, minInputValue, maxInputValue)
	// 	// inputs[0] = -inputs[0]
	// 	n3.calcOutput()
	// 	const actual = n3.output
	// 	const expected = calcExpected(inputs)
	// 	const linearError = actual - expected
	// 	const error = (actual - expected) ** 2
	//
	// 	assert.notOk(Number.isNaN(actual))
	//
	// 	// noinspection PointlessArithmeticExpressionJS
	// 	const dE_do_j = 2 * (actual - expected) / 1
	// 	n3.clear_dE_Dw()
	// 	const sum_sqr_dE_Dw = n3.calc_dE_Dw(dE_do_j)
	//
	// 	if (sum_sqr_dE_Dw !== 0) {
	// 		iteration_sum_sqr_dE_Dw = n
	// 	}
	// 	if (Math.abs(actual - expected) > minError) {
	// 		iteration_error = n
	// 	}
	//
	// 	const now = Date.now()
	// 	if (
	// 		n - iteration_sum_sqr_dE_Dw > 1000 || n - iteration_error > 1000
	// 		|| n < 10 || now - time > 1000 || n === countIterations - 1
	// 	) {
	// 		time = now
	// 		console.log(`${n}: ${
	// 			linearError
	// 		}\r\n---weights---\r\ninput => [n11, n12]: ${
	// 			[n11, n12].map(o => o.weights[0]).join(', ')
	// 		}\r\nn11 => [n21, n22, n23]: ${
	// 			[n21, n22, n23].map(o => o.weights[0]).join(', ')
	// 		}\r\nn12 => [n21, n22, n23]: ${
	// 			[n21, n22, n23].map(o => o.weights[1]).join(', ')
	// 		}\r\n[n21, n22, n23] => n3: ${
	// 			n3.weights.join(', ')
	// 		}\r\n---inputs---\r\n[n11, n12]: ${
	// 			[n11, n12].map(o => o.input).join(', ')
	// 		}\r\nn11 => [n21, n22, n23]: ${
	// 			[n21, n22, n23].map(o => o.input).join(', ')
	// 		}\r\nn3: ${
	// 			n3.input
	// 		}\r\n---outputs---\r\n[n11, n12]: ${
	// 			[n11, n12].map(o => o.output).join(', ')
	// 		}\r\nn11 => [n21, n22, n23]: ${
	// 			[n21, n22, n23].map(o => o.output).join(', ')
	// 		}\r\nn3: ${
	// 			n3.output
	// 		}`)
	// 	}
	//
	// 	if (n - iteration_sum_sqr_dE_Dw > 1000 || n - iteration_error > 1000) {
	// 		break
	// 	}
	//
	// 	if (sum_sqr_dE_Dw !== 0) {
	// 		n3.changeWeights(learningRate * error / sum_sqr_dE_Dw)
	// 	}
	//
	// 	prev_sum_sqr_dE_Dw = sum_sqr_dE_Dw
	// 	prev_linearError = linearError
	// }
}
