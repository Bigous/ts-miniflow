/// <reference path='../node_modules/@types/mocha/index.d.ts' />

import * as chai from "chai";
import * as NDArray from "ndarray";
import ops = require("ndarray-ops");
import { Input, Linear, Node } from "../src";
import { forwardAndBackward, topologicalSort } from "../src/heplers";

const expect = chai.expect;

// tslint:disable:no-unused-expression

describe("Linear", () => {
  const X = new Input();
  const W = new Input();
  const b = new Input();
  const X_ = NDArray([-1, -2, -1, -2], [2, 2]);
  const W_ = NDArray([2, -3, 2, -3], [2, 2]);
  const B_ = NDArray([-3, -5], [1, 2]);
  const X1_ = NDArray([-1, -2, -1, -2], [2, 2]);
  const W1_ = NDArray([2, -3, 2, -3], [2, 2]);
  const B1_ = NDArray([-3, -5], [1, 2]);
  const f = new Linear(X, W, b);
  const feedDict = new Map<Node, any>([[X, X_], [W, W_], [b, B_]]);
  const graph = topologicalSort(feedDict);
  forwardAndBackward(graph);

  it("should verify inputs", () => {
    expect(f.X).to.be.deep.equals(X1_);
    expect(f.W).to.be.deep.equals(W1_);
    expect(f.b).to.be.deep.equals(B1_);
  });

  it("should verify forward propagation", () => {
    const output = f.value as NDArray;
    const expected = NDArray([-9, 4, -9, 4], [2, 2]);
    expect(ops.equals(output, expected)).to.be.true;
  });
});
