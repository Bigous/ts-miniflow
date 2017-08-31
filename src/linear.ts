import * as NDArray from "ndarray";
import * as ops from "ndarray-ops";

import { add, addeq, mul, sigmoid, sumColumns } from "./mathFuncs";
import { Node } from "./node";

export class Linear extends Node {
  public value: any;

  constructor(X: Node, W: Node, b: Node) {
    super([X, W, b]);
  }

  get X(): NDArray {
    return this.inboundNodes[0].value as NDArray;
  }

  get W(): NDArray {
    return this.inboundNodes[1].value as NDArray;
  }

  get b(): NDArray {
    return this.inboundNodes[2].value as NDArray;
  }

  public forward(): void {
    const X = this.X;
    const W = this.W;
    const b = this.b;
    if (!this.value) {
      this.value = NDArray(new Float32Array(X.shape[0] * W.shape[1]), [
        X.shape[0],
        W.shape[1]
      ]);
    }
    // self.value = np.dot(X, W) + b
    addeq(mul(this.value, X, W), b);
  }

  public backward(): void {
    for (const n of this.inboundNodes) {
      const m = n.value as NDArray;
      this.gradients.set(n, NDArray(new Float32Array(m.size), m.shape));
    }
    const X = this.X;
    const W = this.W;
    const b = this.b;
    // const tmp2 = NDArray(new Float32Array(g2.size), g2.shape);
    for (const n of this.outboundNodes) {
      // # Get the partial of the cost with respect to this node.
      // grad_cost = n.gradients[self]
      const gradCost = n.gradGet(this);

      let g0 = this.gradGet(this.inboundNodes[0]);
      if (!g0) {
        const shape = [gradCost.shape[0], W.transpose(1, 0).shape[1]];
        const size = shape[0] * shape[1];
        g0 = NDArray(new Float32Array(size), shape);
        this.gradients.set(this.inboundNodes[0], g0);
      }
      let g1 = this.gradGet(this.inboundNodes[1]);
      if (!g1) {
        const shape = [X.transpose(1, 0).shape[0], gradCost.shape[1]];
        const size = shape[0] * shape[1];
        g1 = NDArray(new Float32Array(size), shape);
        this.gradients.set(this.inboundNodes[1], g1);
      }
      let g2 = this.gradGet(this.inboundNodes[2]);
      if (!g2) {
        const shape = [1, gradCost.shape[1]];
        const size = shape[0] * shape[1];
        g2 = NDArray(new Float32Array(size), shape);
        this.gradients.set(this.inboundNodes[2], g2);
      }
      const tmp0 = NDArray(new Float32Array(g0.size), g0.shape);
      const tmp1 = NDArray(new Float32Array(g1.size), g1.shape);

      // # Set the partial of the loss with respect to this node's inputs.
      // self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
      addeq(g0, mul(tmp0, gradCost, W.transpose(1, 0)));
      // this.gradients.set(
      //   this.inboundNodes[0],
      //   add(tmp0, g0, mul(tmp0, gradCost, W.transpose(1, 0)))
      // );

      // # Set the partial of the loss with respect to this node's weights.
      // self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
      addeq(g1, mul(tmp1, X.transpose(1, 0), gradCost));
      // this.gradients.set(
      //   this.inboundNodes[1],
      //   add(tmp1, g1, mul(tmp1, X.transpose(1, 0), gradCost))
      // );

      // # Set the partial of the loss with respect to this node's bias.
      // self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)

      addeq(g2, sumColumns(gradCost));
      // this.gradients.set(
      //   this.inboundNodes[2],
      //   add(tmp2, g2, sumColumns(gradCost))
      // );
    }
  }
}
