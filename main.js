import * as tf from "@tensorflow/tfjs-node";
import * as readline from "node:readline";

const size = 512;

const input = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [2] }));
model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

let _data = generateTestData();

const _input = tf.tensor(_data[0], [size, 2]);
const _answers = tf.tensor(_data[1], [size, 1]);

await model.fit(_input, _answers, { epochs: 250 });

getInput();

function getInput() {
    input.question("Please input your calculation!", (_input) => {
        let _nums = _input.split("+");
        console.log(_input + " = " + model.predict(tf.tensor2d([parseInt(_nums[0].trim()), parseInt(_nums[1].trim())], [1, 2])).dataSync()[0]);

        getInput();
    });
}

function generateTestData() {
    let _input = [];
    let _answers = [];

    for(let i = 0; i < size; i++) {
        let _one = (Math.random() * 10);
        let _two = (Math.random() * 10);
        let _ans = _one + _two;

        _input[_input.length] = [_one, _two];
        _answers[_answers.length] = _ans;
    }

    return [_input, _answers];
}