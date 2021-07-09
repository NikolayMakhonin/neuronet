import {TNeuronFunc, TNeuronInput} from "./contracts";
import {fixInfinity} from "./helpers";

export class Neuron {
	readonly func: TNeuronFunc
	readonly inputs: TNeuronInput
	readonly weights: number[]
	readonly prevWeights: number[]
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
		this.prevWeights = []
		const len = inputs.length

		if (this.weights.length !== len) {
			throw new Error(`weights.length (${this.weights.length}) !== input.length (${len})`)
		}
		if (this.fixedWeights && this.fixedWeights.length !== len) {
			throw new Error(`fixedWeights.length (${this.fixedWeights.length}) !== input.length (${len})`)
		}

		for (let i = 0; i < len; i++) {
			this.dE_dw[i] = 0
			this.prevWeights[i] = this.weights[i]
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
		}
	}

	/**
	 * @param dDk_dyj = 2 * (actual - expected)
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
			sum_sqr_dE_Dw -= this.dE_dw[j] * this.dE_dw[j]
			this.dE_dw[j] += dEk_dwij
			sum_sqr_dE_Dw += this.dE_dw[j] * this.dE_dw[j]
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

	calcSumSqr_dE_Dw() {
		let sum_sqr_dE_Dw = 0
		for (let j = 0, len = this.weights.length; j < len; j++) {
			sum_sqr_dE_Dw += this.dE_dw[j] * this.dE_dw[j]
		}
		return sum_sqr_dE_Dw
	}

	changeWeights(dw: number, momentRate?: number) {
		for (let i = 0, len = this.weights.length; i < len; i++) {
			if (this.fixedWeights && this.fixedWeights[i]) {
				continue
			}
			let weight = this.weights[i]
			const prevDw = this.weights[i] - this.prevWeights[i]
			this.prevWeights[i] = weight

			weight = weight - dw * this.dE_dw[i]
			if (momentRate) {
				weight += momentRate * prevDw
			}

			this.weights[i] = fixInfinity(weight)
		}
	}
}
