/// <reference path='../node_modules/@types/mocha/index.d.ts' />

import * as bm from "bluemath";
import * as chai from "chai";
import { Input, Linear, Node } from "../src";
import { forwardAndBackward, topologicalSort } from "../src/heplers";

const expect = chai.expect;

describe("Linear", () => {
  it("should output [[-9, 4],[-9, 4]]", () => {
    const X = new Input();
    const W = new Input();
    const b = new Input();
    const X_ = new bm.NDArray([[-1, -2], [-1, -2]]);
    const W_ = new bm.NDArray([[2, -3], [2, -3]]);
    const B_ = new bm.NDArray([-3, -5]);
    const f = new Linear(X, W, b);
    const feedDict = new Map<Node, any>([[X, X_], [W, W_], [b, B_]]);
    const graph = topologicalSort(feedDict);
    forwardAndBackward(graph);
    const output = f.value as bm.NDArray;
    const expected = new bm.NDArray([[-9, 4], [-9, 4]]);
    expect(output).to.be.equals(expected, "Was?");
  });
});
