import js from "@eslint/js";

export default [
	{
		ignores: [
			"**/node_modules/",
			".git/",
			"dist/",
			"**/*-inline.js",
			"lib/core/web-worker-inline-native.js",
			"lib/core/web-worker-inline-wasm.js",
			"lib/core/streams/zlib-js/zlib-streams.min.js",
			"tests/vendor/*.js",
			"index.cjs",
			"index.min.js",
			"index-native.cjs",
			"index-native.min.js"
		]
	},
	js.configs.recommended,
	{
		languageOptions: {
			ecmaVersion: 2020,
			sourceType: "module",
			globals: {
				console: "readonly"
			}
		},
		rules: {
			"indent": [
				"error",
				"tab",
				{
					"SwitchCase": 1
				}
			],
			"linebreak-style": [
				"error",
				"unix"
			],
			"quotes": [
				"error",
				"double"
			],
			"semi": [
				"error",
				"always"
			],
			"no-console": [
				"warn"
			]
		}
	}
];