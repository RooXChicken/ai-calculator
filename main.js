import * as tf from "@tensorflow/tfjs-node";
import * as readline from "node:readline";

const size = 1024*2;

const input = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [3] }));
model.compile({ loss: tf.losses.absoluteDifference, optimizer: "adam" });

let _data = generateTestData();

const _input = tf.tensor(_data[0], [size, 3]);
const _answers = tf.tensor(_data[1], [size, 1]);

await model.fit(_input, _answers, { epochs: 64 });

getInput();

function getInput() {
    input.question("Please input your calculation! ", (_input) => {
        let _nums = _input.split(RegExp("(\\+|-|\\*|\\/)"));
        let _operation = (_nums[1] == "+") ? 0 : (_nums[1] == "-") ? 1 : (_nums[1] == "*") ? 2 : 3;

        let _in = [parseInt(_nums[0].trim()), parseInt(_nums[2].trim()), _operation];
        console.log(_in);
        let _tensor = tf.tensor(_in, [1, 3]);
        console.log(_input + " = " + model.predict(_tensor).dataSync()[0]);

        getInput();
    });
}

function generateTestData() {
    let _input = [];
    let _answers = [];

    for(let i = 0; i < size; i++) {
        let _one = (Math.random() * 100);
        let _two = (Math.random() * 100);
        let _operation = Math.floor(Math.random()*3);

        let _ans;
        switch(_operation) {
            case 0: _ans = _one + _two; break;
            case 1: _ans = _one - _two; break;
            case 2: _ans = _one * _two; break;
            case 3: _ans = _one / _two; break;
        }

        _input[_input.length] = [_one, _two, _operation];
        _answers[_answers.length] = _ans;
    }

    return [_input, _answers];
}

// function loss(_input, _output) {
//     _input.print();
//     _output.print();

//     return 0;
// }