describe('neuron', function () {
	this.timeout(120000)

	const infinity = 1e10

	function fixInfinity(value: number) {
		if (value >= infinity) {
			return 1e10
		}
		if (value <= -infinity) {
			return -1e10
		}
		if (Number.isNaN(value)) {
			assert.fail()
		}
		return value
	}

	class Neuron {
		constructor(
			func: IFunc,
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

		func: IFunc

		inputs: Array<Neuron|number>
		weights: number[]
		dE_dw: number[]

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

		input: number
		output: number

		calcOutput(): void {
			this._calcInput()
			this.output = fixInfinity(this.func.func(this.input))
		}

		clear_dE_Dw(): void {
			for (let i = 0, len = this.weights.length; i < len; i++) {
				const weight = this.weights[i]
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

	interface IFunc {
		func: (input: number) => number
		derivative: (input: number) => number
	}

	const funcs = {
		x: {
			func: o => o,
			derivative: o => 1,
		},
		sign: {
			func: o => o >= 0 ? 1 : -1,
			derivative: o => Infinity,
		},
		ReLU: {
			func: o => o >= 0 ? o : 0,
			derivative: o => o >= 0 ? 1 : 0,
		},
		sigmoid: {
			func: o => 1 / (1 + Math.E**(-o)),
			derivative(o) {
				return this.func(o) * (1 - this.func(o))
			},
		},
		'1': {
			func: o => 1,
			derivative: o => 0,
		},
		exp: {
			func: o => Math.E ** o,
			derivative(o) {
				return this.func(o)
			},
		},
		ln: {
			func: o => o === 0
				? -Infinity
				: Math.log(Math.abs(o)),
			derivative: o => o === 0
				? Infinity
				: 1 / Math.abs(o),
		},
		sin: {
			func: o => Math.sin(o),
			derivative: o => Math.cos(o),
		},
	}

	it('base', async function() {
		const inputs = [1]

		const n11 = new Neuron(
			funcs.x,
			inputs,
			[null],
		)

		const n12 = new Neuron(
			funcs.ln,
			inputs,
			[null],
		)

		const n21 = new Neuron(
			funcs['1'],
			[n11, n12],
			[null, null],
		)

		const n22 = new Neuron(
			funcs.x,
			[n11, n12],
			[1, 1],
		)

		const n23 = new Neuron(
			funcs.exp,
			[n11, n12],
			[1, 1],
		)

		const n24 = new Neuron(
			funcs.sin,
			[n11, n12],
			[1, 1],
		)

		const n3 = new Neuron(
			funcs.x,
			[n21, n22, n23],
			[1, 1, 1],
		)

		// tslint:disable-next-line:no-shadowed-variable
		function calcExpected(inputs: number[]) {
			return inputs[0] ** 2
			// return Math.log(inputs[0])
		}

		// tslint:disable-next-line:no-shadowed-variable
		function fillInputsRandom(inputs: number[], minValue: number, maxValue: number) {
			for (let i = 0, len = inputs.length; i < len; i++) {
				inputs[i] = Math.random() * (maxValue - minValue) + minValue
			}
		}

		const minInputValue = -2 // 1e-10
		const maxInputValue = 4 // 10
		const learningRate = 1
		const countIterations = 100000000
		const minError = 1e-13
		let time = Date.now()
		let prev_sum_sqr_dE_Dw
		let prev_linearError
		let iteration_sum_sqr_dE_Dw = 0
		let iteration_error = 0

		for (let n = 0; n < countIterations; n++) {
			fillInputsRandom(inputs, minInputValue, maxInputValue)
			// inputs[0] = -inputs[0]
			n3.calcOutput()
			const actual = n3.output
			const expected = calcExpected(inputs)
			const linearError = actual - expected
			const error = (actual - expected) ** 2

			assert.notOk(Number.isNaN(actual))

			// noinspection PointlessArithmeticExpressionJS
			const dE_do_j = 2 * (actual - expected) / 1
			n3.clear_dE_Dw()
			const sum_sqr_dE_Dw = n3.calc_dE_Dw(dE_do_j)

			if (sum_sqr_dE_Dw !== 0) {
				iteration_sum_sqr_dE_Dw = n
			}
			if (Math.abs(actual - expected) > minError) {
				iteration_error = n
			}

			const now = Date.now()
			if (
				n - iteration_sum_sqr_dE_Dw > 1000 || n - iteration_error > 1000
				|| n < 10 || now - time > 1000 || n === countIterations - 1
			) {
				time = now
				console.log(`${n}: ${
					linearError
				}\r\n---weights---\r\ninput => [n11, n12]: ${
					[n11, n12].map(o => o.weights[0]).join(', ')
				}\r\nn11 => [n21, n22, n23]: ${
					[n21, n22, n23].map(o => o.weights[0]).join(', ')
				}\r\nn12 => [n21, n22, n23]: ${
					[n21, n22, n23].map(o => o.weights[1]).join(', ')
				}\r\n[n21, n22, n23] => n3: ${
					n3.weights.join(', ')
				}\r\n---inputs---\r\n[n11, n12]: ${
					[n11, n12].map(o => o.input).join(', ')
				}\r\nn11 => [n21, n22, n23]: ${
					[n21, n22, n23].map(o => o.input).join(', ')
				}\r\nn3: ${
					n3.input
				}\r\n---outputs---\r\n[n11, n12]: ${
					[n11, n12].map(o => o.output).join(', ')
				}\r\nn11 => [n21, n22, n23]: ${
					[n21, n22, n23].map(o => o.output).join(', ')
				}\r\nn3: ${
					n3.output
				}`)
			}

			if (n - iteration_sum_sqr_dE_Dw > 1000 || n - iteration_error > 1000) {
				break
			}

			if (sum_sqr_dE_Dw !== 0) {
				n3.changeWeights(learningRate * error / sum_sqr_dE_Dw)
			}

			prev_sum_sqr_dE_Dw = sum_sqr_dE_Dw
			prev_linearError = linearError
		}

		console.log(`searchIterations = ${Math.max(iteration_sum_sqr_dE_Dw, iteration_error)}`)
	})
})
