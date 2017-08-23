import * as bm from "bluemath";
import { sigmoid } from "./mathFuncs";
import { Node } from "./node";

export class MSE extends Node {
  public m: number;
  public diff: bm.NDArray;
  constructor(y: Node, a: Node) {
    super([ y, a ]);
  }

  public forward(): void {
    const y = this.inboundNodes[0].value as bm.NDArray;
    y.reshape([y.shape[0] -1, y.shape[1] + 1]);
    const a = this.inboundNodes[1].value as bm.NDArray;
    a.reshape([a.shape[0]-1, a.shape[1]+1]);

    this.m = y.shape[0];
    this.diff = bm.sub(y, a) as bm.NDArray;
    this.value = 0
    for(let i = 0; i < this.diff.shape[0]; i++) {
      this.value += (this.diff.get(i,0) as number) ** 2;
    }
    this.value /= this.diff.shape[0];
  }

  public backward(): void {
    this.gradients.set(
      this.inboundNodes[0],
      bm.mul(this.diff, 2 / this.m)
    );
    this.gradients.set(
      this.inboundNodes[1],
      bm.mul(this.diff, -2 / this.m)
    );
  }
}
