import * as NDArray from "ndarray";
import * as ops from "ndarray-ops";

import { add, addeq, mul, sigmoid, sumColumns } from "./mathFuncs";
import { Node } from "./node";

export class Input extends Node {
  constructor() {
    super();
  }

  public forward(): void {
    // nothing... realy...
  }

  public backward(): void {
    // # The key, `self`, is reference to this object.
    // self.gradients = {self: 0}
    this.gradients.set(this, 0);
    // # Weights and bias may be inputs, so you need to sum
    // # the gradient from output gradients.
    // for n in self.outbound_nodes:
    //     grad_cost = n.gradients[self]
    //     self.gradients[self] += grad_cost * 1
    for (const n of this.outboundNodes) {
      const gradCost = n.gradGet(this);
      // const tmp1 = NDArray(new Float32Array(gradCost.size), gradCost.shape);
      const g = this.gradGet(this);
      // this.gradients.set(this, add(tmp1, g, gradCost));
      if (g) {
        addeq(g, gradCost);
      } else {
        this.gradients.set(this, gradCost);
      }
    }
  }
}
