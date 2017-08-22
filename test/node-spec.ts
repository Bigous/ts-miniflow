/// <reference path='../node_modules/@types/mocha/index.d.ts' />

import { Node } from "../src/node";
import * as chai from "chai";

const expect = chai.expect;

describe("Node", () => {
  it("should greet with message", () => {
    const greeter = new Node("friend");
    expect(greeter.greet()).to.equal("Bonjour, friend!");
  });
});
