import * as bm from "bluemath";
import { sigmoid } from "./mathFuncs";
import { Node } from "./node";

export class Sigmoid extends Node {
  constructor(node: Node) {
    super([node]);
  }

  public forward(): void {
    this.value = sigmoid(this.inboundNodes[0].value as number);
  }

  public backward(): void {
    for (const n of this.inboundNodes) {
      this.gradients.set(n, bm.zeros(n.value as any));
    }
    for (const n of this.outboundNodes) {
      const gradCost = n.gradients.get(this);
      const s = this.value as number;
      this.gradients.set(
        this.inboundNodes[0],
        bm.add(
          this.gradients.get(this.inboundNodes[0]) as any,
          bm.mul(gradCost as any, s * (1 - s))
        )
      );
    }
  }
}
