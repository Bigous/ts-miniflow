import * as NDArray from "ndarray";
import ops = require("ndarray-ops");
import { addeq, mul, sigmoida } from "./mathFuncs";
import { Node } from "./node";

export class Sigmoid extends Node {
  constructor(node: Node) {
    super([node]);
  }

  public forward(): void {
    const a = this.inboundNodes[0].value as NDArray;
    if (!this.value) {
      this.value = NDArray(new Float32Array(a.size), a.shape);
    }
    sigmoida(this.value as NDArray, a);
  }

  public backward(): void {
    for (const n of this.inboundNodes) {
      const m = n.value as NDArray;
      this.gradients.set(n, NDArray(new Float32Array(m.size), m.shape));
    }
    const s = this.value as NDArray;
    const tmp = NDArray(new Float32Array(s.size), s.shape);
    for (const n of this.outboundNodes) {
      const gradCost = n.gradients.get(this) as NDArray;
      const tmp1 = NDArray(new Float32Array(s.size), s.shape);
      let g0 = this.gradGet(this.inboundNodes[0]);
      if (!g0) {
        g0 = NDArray(new Float32Array(tmp1.size), tmp1.shape);
        this.gradients.set(this.inboundNodes[0], g0);
      }
      addeq(
        g0,
        ops.mul(
          tmp1,
          ops.muleq(ops.addseq(ops.muls(tmp, s, -1), 1), s),
          gradCost.transpose(1, 0)
        )
      );
    }
  }
}
