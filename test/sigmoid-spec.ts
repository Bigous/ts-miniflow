/// <reference path='../node_modules/@types/mocha/index.d.ts' />

import * as chai from "chai";
import * as NDArray from "ndarray";
import ops = require("ndarray-ops");
import { Input, Linear, Node, Sigmoid } from "../src";
import { forwardAndBackward, topologicalSort } from "../src/heplers";

const expect = chai.expect;

// tslint:disable:no-unused-expression

describe("Sigmoid", () => {
  const X = new Input();
  const W = new Input();
  const b = new Input();
  const X_ = NDArray([-1, -2, -1, -2], [2, 2]);
  const W_ = NDArray([2, -3, 2, -3], [2, 2]);
  const B_ = NDArray([-3, -5], [1, 2]);
  const f = new Linear(X, W, b);
  const s = new Sigmoid(f);
  const feedDict = new Map<Node, any>([[X, X_], [W, W_], [b, B_]]);
  const graph = topologicalSort(feedDict);
  forwardAndBackward(graph);

  it("should verify value", () => {
    const expected = NDArray(
      [
        1.2339458044152707e-4,
        9.820137619972229e-1,
        1.2339458044152707e-4,
        9.820137619972229e-1
      ],
      [2, 2]
    );
    expect(ops.equals(s.value, expected)).to.be.true;
  });
});
