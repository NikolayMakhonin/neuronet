import {INeuronFunc, TNeuronInput} from "./contracts";
import {fixInfinity} from "./helpers";

export class Neuron {
	readonly func: INeuronFunc
	readonly inputs: TNeuronInput
	readonly weights: number[]
	readonly dE_dw: number[]

	input: number
	output: number

	constructor(
		func: INeuronFunc,
		inputs: Array<Neuron|number>,
		weights: number[],
	) {
		this.func = func
		this.inputs = inputs
		this.weights = weights
		this.dE_dw = []
		for (let i = 0, len = inputs.length; i < len; i++) {
			this.dE_dw[i] = 0
		}
	}

	private _calcInput() {
		let sum = 0
		for (let i = 0, len = this.weights.length; i < len; i++) {
			let weight = this.weights[i]
			if (weight == null) {
				weight = 1
			}
			const input = this.inputs[i]
			let inputValue: number
			if (typeof input === 'number') {
				inputValue = input
			} else {
				input.calcOutput()
				inputValue = input.output
			}

			sum += weight * inputValue
		}
		this.input = fixInfinity(sum)
	}

	calcOutput(): void {
		this._calcInput()
		this.output = fixInfinity(this.func.func(this.input))
	}

	clear_dE_Dw(): void {
		for (let i = 0, len = this.weights.length; i < len; i++) {
			this.dE_dw[i] = 0
			const input = this.inputs[i]
			if (typeof input !== 'number') {
				input.clear_dE_Dw()
			}
		}
	}

	/**
	 * @param dE_do_j = 2 * (actual - expected) * learningRate
	 * @return sum_sqr_dE_Dw
	 */
	calc_dE_Dw(dE_do_j: number): number {
		dE_do_j = fixInfinity(dE_do_j)
		const dF_dS_j = this.func.derivative(this.input)
		let sum_sqr_dE_Dw = 0
		for (let i = 0, len = this.weights.length; i < len; i++) {
			const w_ij = this.weights[i]
			if (w_ij == null) {
				continue
			}
			const input = this.inputs[i]
			const o_i = typeof input === 'number'
				? input
				: input.output
			const dE_dw_ij = dE_do_j * dF_dS_j * o_i
			this.dE_dw[i] += dE_dw_ij
			sum_sqr_dE_Dw += dE_dw_ij * dE_dw_ij
			// if (dE_dw_ij !== 0) {
			// 	this.weights[i] = fixInfinity(w_ij - dE_dw_ij)
			// }
			if (typeof input !== 'number') {
				const dE_do_i = dE_do_j * dF_dS_j * w_ij
				sum_sqr_dE_Dw += input.calc_dE_Dw(dE_do_i)
			}
		}

		return fixInfinity(sum_sqr_dE_Dw)
	}

	changeWeights(learningRate: number) {
		for (let i = 0, len = this.weights.length; i < len; i++) {
			const weight = this.weights[i]
			if (weight == null) {
				continue
			}
			this.weights[i] = fixInfinity(weight - learningRate * this.dE_dw[i])
			const input = this.inputs[i]
			if (typeof input !== 'number') {
				input.changeWeights(learningRate)
			}
		}
	}
}
