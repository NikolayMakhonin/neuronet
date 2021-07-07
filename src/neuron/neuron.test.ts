import assert from "assert";
import {Neuron} from "./neuron";
import {funcs} from "./funcs";

describe('Neuron', function() {
	it('base', function() {
		const inputs: (Neuron|number)[] = [2, 3, 4]
		const weights = [5, 6, 7]
		const neuron = new Neuron(funcs.sin, inputs, weights)

		neuron.calcOutput()
		assert.strictEqual(neuron.output, funcs.sin.func(2 * 5 + 3 * 6 + 4 * 7))

		inputs[0] = 8
		inputs[1] = 9
		inputs[2] = 10
		weights[0] = 11
		weights[1] = 12
		weights[2] = 13

		assert.strictEqual(neuron.output, funcs.sin.func(2 * 5 + 3 * 6 + 4 * 7))
		neuron.calcOutput()
		assert.strictEqual(neuron.output, funcs.sin.func(8 * 11 + 9 * 12 + 10 * 13))

		inputs[0] = new Neuron(funcs.x, [14], [1])
		inputs[1] = new Neuron(funcs.x, [15], [1])
		inputs[2] = new Neuron(funcs.x, [16], [1])

		assert.strictEqual(neuron.output, funcs.sin.func(8 * 11 + 9 * 12 + 10 * 13))
		neuron.calcOutput()
		assert.strictEqual(neuron.output, funcs.sin.func(14 * 11 + 15 * 12 + 16 * 13))
	})

	it('chain', function() {
		const n1 = new Neuron(funcs.sin, [2], [3])
		const n2 = new Neuron(funcs.exp, [n1], [4])
		const n3 = new Neuron(funcs.sigmoid, [n2], [5])

		n3.calcOutput()
		assert.strictEqual(n3.output, funcs.sigmoid.func(funcs.exp.func(funcs.sin.func(2 * 3) * 4) * 5))
		assert.strictEqual(n2.output, funcs.exp.func(funcs.sin.func(2 * 3) * 4))
		assert.strictEqual(n1.output, funcs.sin.func(2 * 3))
	})
})
