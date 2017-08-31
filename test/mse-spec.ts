/// <reference path='../node_modules/@types/mocha/index.d.ts' />

import * as chai from "chai";
import * as NDArray from "ndarray";
import ops = require("ndarray-ops");
import { Input, Linear, MSE, Node, Sigmoid } from "../src";
import { forwardAndBackward, topologicalSort } from "../src/heplers";

const expect = chai.expect;

// tslint:disable:no-unused-expression
// tslint:disable:no-console

describe("MSE", () => {
  // tslint:disable-next-line:curly
  // if (1 === 1) return;
  const X = new Input();
  const W = new Input();
  const b = new Input();
  const y = new Input();
  const f = new Linear(X, W, b);
  const a = new Sigmoid(f);
  const cost = new MSE(y, a);
  const X_ = NDArray([-1, -2, -1, -2], [2, 2]);
  const W_ = NDArray([2.0, 3.0], [2, 1]);
  const B_ = NDArray([-3], [1, 1]);
  const Y_ = NDArray([1, 2], [1, 2]);
  const feedDict = new Map<Node, any>([[X, X_], [W, W_], [b, B_], [y, Y_]]);
  const graph = topologicalSort(feedDict);

  forwardAndBackward(graph);

  it("should verify final value", () => {
    // const expected = 2.49994989601;
    const expected = 2.5;
    expect(cost.value).to.be.equal(expected);
  });
});
