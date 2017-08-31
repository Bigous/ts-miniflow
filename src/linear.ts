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

  get b(): number | NDArray {
    return this.inboundNodes[2].value;
  }

  public forward(): void {
    const X = this.X;
    const W = this.W;
    const b = this.b;
    const tmp1 = NDArray(new Float32Array(X.size), X.shape);
    mul(tmp1, X, W);
    this.value = addeq(tmp1, b as NDArray);
  }

  public backward(): void {
    for (const n of this.inboundNodes) {
      const m = n.value as NDArray;
      this.gradients.set(n, NDArray(new Float32Array(m.size), m.shape));
    }
    const X = this.X;
    const W = this.W;
    const b = this.b;
    for (const n of this.outboundNodes) {
      const g0 = this.gradGet(this.inboundNodes[0]);
      const g1 = this.gradGet(this.inboundNodes[1]);
      const g2 = this.gradGet(this.inboundNodes[2]);
      // # Get the partial of the cost with respect to this node.
      // grad_cost = n.gradients[self]
      const gradCost = n.gradGet(this);

      // # Set the partial of the loss with respect to this node's inputs.
      // self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
      const tmp0 = NDArray(new Float32Array(g0.size), g0.shape);
      this.gradients.set(
        this.inboundNodes[0],
        add(tmp0, g0, mul(tmp0, gradCost, W.transpose(1, 0)))
      );

      // # Set the partial of the loss with respect to this node's weights.
      // self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
      const tmp1 = NDArray(new Float32Array(g1.size), g1.shape);
      this.gradients.set(
        this.inboundNodes[1],
        add(tmp1, g1, mul(tmp1, X.transpose(1, 0), gradCost))
      );

      // # Set the partial of the loss with respect to this node's bias.
      // self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)
      const tmp2 = NDArray(new Float32Array(g2.size), g2.shape);
      this.gradients.set(
        this.inboundNodes[2],
        add(tmp2, g2, sumColumns(gradCost))
      );
    }
  }
}
