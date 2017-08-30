import * as bm from "bluemath";

import { inv, sigmoid, sumColumns } from "./mathFuncs";
import { Node } from "./node";

export class Linear extends Node {
  public value: any;

  constructor(X: Node, W: Node, b: Node) {
    super([X, W, b]);
  }

  get X(): bm.NDArray {
    return this.inboundNodes[0].value as bm.NDArray;
  }

  get Y(): bm.NDArray {
    return this.inboundNodes[1].value as bm.NDArray;
  }

  get b(): any {
    return this.inboundNodes[2].value;
  }

  public forward(): void {
    const X = this.X;
    const W = this.Y;
    const b = this.b;
    // tslint:disable-next-line:no-console
    console.log("Aeee X:");
    // tslint:disable-next-line:no-console
    console.log(X);
    // tslint:disable-next-line:no-console
    console.log(W);
    // tslint:disable-next-line:no-console
    console.log(b);
    // tslint:disable-next-line:no-console
    console.log("MatMul:");
    // tslint:disable-next-line:no-console
    console.log(bm.linalg.matmul(X, W));
    this.value = bm.add(bm.linalg.matmul(X, W), b);
    // tslint:disable-next-line:no-console
    console.log("Deu:");
    // tslint:disable-next-line:no-console
    console.log(this.value);
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
