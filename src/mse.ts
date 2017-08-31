import * as NDArray from "ndarray";
import ops = require("ndarray-ops");
import { sigmoid } from "./mathFuncs";
import { Node } from "./node";

export class MSE extends Node {
  public m: number;
  public diff: NDArray;
  constructor(y: Node, a: Node) {
    super([y, a]);
  }

  get y(): NDArray {
    return this.inboundNodes[0].value as NDArray;
  }

  get a(): NDArray {
    return this.inboundNodes[1].value as NDArray;
  }

  public forward(): void {
    const y = this.y;
    // TODO: implement reshape
    // y.reshape([y.shape[0] - 1, y.shape[1] + 1]);
    const a = this.a;
    // a.reshape([a.shape[0] - 1, a.shape[1] + 1]);

    this.m = y.shape[0];
    if (!this.diff) {
      this.diff = NDArray(new Float32Array(y.size), y.shape);
    }

    ops.subeq(this.diff, y, a);
    this.value = 0;
    for (const v of this.diff.data) {
      this.value += v * v;
    }
    this.value /= this.diff.size;

    // const tmp = ops.assign(NDArray(new Float32Array(y.size), y.shape), this.diff);
    // this.value = ops.sum(ops.muleq(tmp,tmp)) / tmp.size;

    // for (let i = 0; i < this.diff.shape[0]; i++) {
    //   this.value += (this.diff.get(i, 0) as number) ** 2;
    // }
    // this.value /= this.diff.shape[0];
  }

  public backward(): void {
    // self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
    // self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff
    let g0 = this.gradGet(this.inboundNodes[0]);
    if (!g0) {
      g0 = NDArray(new Float32Array(this.diff.size), this.diff.shape);
      this.gradients.set(this.inboundNodes[0], g0);
    }
    let g1 = this.gradGet(this.inboundNodes[1]);
    if (!g1) {
      g1 = NDArray(new Float32Array(this.diff.size), this.diff.shape);
      this.gradients.set(this.inboundNodes[1], g1);
    }
    ops.muls(g0, this.diff, 2 / this.m);
    ops.muls(g1, this.diff, -2 / this.m);
    // this.gradients.set(this.inboundNodes[0], ops.mul(this.diff, 2 / this.m));
    // this.gradients.set(this.inboundNodes[1], ops.mul(this.diff, -2 / this.m));
  }
}
