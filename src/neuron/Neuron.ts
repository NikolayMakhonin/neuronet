import {TNeuronFunc, TNeuronInput} from "./contracts";
import {fixInfinity} from "./helpers";

export class Neuron {
	readonly func: TNeuronFunc
	readonly inputs: TNeuronInput
	readonly weights: number[]
	readonly fixedWeights: boolean[]
	readonly dE_dw: number[]

	input: number
	output: number

	constructor(
		func: TNeuronFunc,
		inputs: Array<Neuron|number>,
		weights: number[],
		fixedWeights?: boolean[],
	) {
		this.func = func
		this.inputs = inputs
		this.weights = weights
		this.fixedWeights = fixedWeights
		this.dE_dw = []
		for (let i = 0, len = inputs.length; i < len; i++) {
			this.dE_dw[i] = 0
		}
	}

	private _calcInput() {
		let sum = 0
		for (let i = 0, len = this.weights.length; i < len; i++) {
			let weight = this.weights[i]
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
	 * @param dDk_dyj = 2 * (actual - expected) * learningRate
	 * @return sum_sqr_dE_Dw
	 */
	calc_dE_Dw(dDk_dyj: number): number {
		dDk_dyj = fixInfinity(dDk_dyj)
		const df_dSi = this.func.derivative(this.input)
		let sum_sqr_dE_Dw = 0
		for (let j = 0, len = this.weights.length; j < len; j++) {
			if (this.fixedWeights && this.fixedWeights[j]) {
				continue
			}
			const w_ij = this.weights[j]
			const input = this.inputs[j]
			const xj = typeof input === 'number'
				? input
				: input.output
			const dyj_dwij = df_dSi * xj
			const dEk_dwij = dDk_dyj * dyj_dwij
			this.dE_dw[j] += dEk_dwij
			sum_sqr_dE_Dw += dEk_dwij * dEk_dwij
			// if (dE_dw_ij !== 0) {
			// 	this.weights[i] = fixInfinity(w_ij - dE_dw_ij)
			// }
			if (typeof input !== 'number') {
				const dDk_dxj = dEk_dwij * w_ij
				sum_sqr_dE_Dw += input.calc_dE_Dw(dDk_dxj)
			}
		}

		return fixInfinity(sum_sqr_dE_Dw)
	}

	changeWeights(learningRate: number) {
		for (let i = 0, len = this.weights.length; i < len; i++) {
			if (this.fixedWeights && this.fixedWeights[i]) {
				continue
			}
			const weight = this.weights[i]
			this.weights[i] = fixInfinity(weight - learningRate * this.dE_dw[i])
			const input = this.inputs[i]
			if (typeof input !== 'number') {
				input.changeWeights(learningRate)
			}
		}
	}
}
