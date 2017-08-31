import * as NDArray from "ndarray";
import ops = require("ndarray-ops");
import { sigmoids } from "./mathFuncs";
import { Node } from "./node";

export class Sigmoid extends Node {
  constructor(node: Node) {
    super([node]);
  }

  public forward(): void {
    this.value = sigmoids(this.inboundNodes[0].value as number);
  }

  public backward(): void {
    for (const n of this.inboundNodes) {
      this.gradients.set(
        n,
        NDArray(new Float32Array(n.value as any).fill(0.0))
      );
    }
    for (const n of this.outboundNodes) {
      const gradCost = n.gradients.get(this);
      const s = this.value as number;
      this.gradients.set(
        this.inboundNodes[0],
        ops.add(
          this.gradients.get(this.inboundNodes[0]) as any,
          ops.mul(gradCost as any, s * (1 - s))
        )
      );
    }
  }
}
