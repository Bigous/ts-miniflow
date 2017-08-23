import * as bm from "bluemath";

import { inv, sigmoid, sumColumns } from "./mathFuncs";
import { Node } from "./node";

export class Linear extends Node {
  constructor(X: Node, W: Node, b: Node) {
    super([X, W, b]);
  }

  public forward(): void {
    const X = this.inboundNodes[0].value;
    const W = this.inboundNodes[1].value;
    const b = this.inboundNodes[2].value;
    this.value = bm.add(bm.mul(X, W), b);
  }

  public backward(): void {
    for (const n of this.inboundNodes) {
      this.gradients.set(n, bm.zeros(n.value as number));
    }
    for (const n of this.outboundNodes) {
      const gradCost = n.gradients.get(this);
      this.gradients.set(
        this.inboundNodes[0],
        bm.add(
          this.gradients.get(this.inboundNodes[0]) as any,
          bm.mul(gradCost as any, inv(this.inboundNodes[1].value as bm.NDArray))
        )
      );
      this.gradients.set(
        this.inboundNodes[1],
        bm.add(
          this.gradients.get(this.inboundNodes[1]) as any,
          bm.mul(inv(this.inboundNodes[0].value as bm.NDArray), gradCost as any)
        )
      );
      this.gradients.set(
        this.inboundNodes[2],
        bm.add(
          this.gradients.get(this.inboundNodes[2]) as any,
          sumColumns(gradCost as bm.NDArray)
        )
      );
    }
  }
}
