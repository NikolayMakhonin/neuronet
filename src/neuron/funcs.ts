import {INeuronFunc} from "./contracts";

export const funcs: {
	[name: string]: INeuronFunc
} = {
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
