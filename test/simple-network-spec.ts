/// <reference path='../node_modules/@types/mocha/index.d.ts' />

import * as chai from "chai";
import * as NDArray from "ndarray";
import ops = require("ndarray-ops");
import { Input, Linear, MSE, Node, Sigmoid } from "../src";
import { forwardAndBackward, topologicalSort } from "../src/heplers";

const expect = chai.expect;

// tslint:disable:no-unused-expression
// tslint:disable:no-console

describe("Back propagation", () => {
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

  console.log("GRAPH:");
  console.log(graph);

  forwardAndBackward(graph);
  const backX = X.gradGet(X);
  const backY = y.gradGet(y);
  const backW = W.gradGet(W);
  const backB = b.gradGet(b);

  console.log("Back X:");
  console.log(backX);
  console.log("Back Y:");
  console.log(backY);
  console.log("Back W:");
  console.log(backW);
  console.log("Back B:");
  console.log(backB);

  it("should verify back propagation", () => {
    const expectedBackX = NDArray(
      [-3.3401728e-5, -5.01025919e-5, -6.68040138e-5, -1.00206021e-4],
      [2, 2]
    );
    const expectedBackY = NDArray([0.9999833, 1.9999833], [2, 1]);
    const expectedBackW = NDArray([5.01028709e-5, 1.00205742e-4], [2, 1]);
    const expectedBackB = NDArray([-5.01028709e-5], [1, 1]);
    expect(backX, "back X undefined").to.not.be.undefined;
    expect(backY, "back Y undefined").to.not.be.undefined;
    expect(backW, "back W undefined").to.not.be.undefined;
    expect(backB, "back B undefined").to.not.be.undefined;
    expect(ops.equals(backX, expectedBackX), "back X").to.be.true;
    expect(ops.equals(backY, expectedBackY), "back Y").to.be.true;
    expect(ops.equals(backW, expectedBackW), "back W").to.be.true;
    expect(ops.equals(backB, expectedBackB), "back B").to.be.true;
  });
});
