import { Node } from "./node";

export class Input extends Node {
  constructor() {
    super();
  }

  public forward(): void {
    // nothing... realy...
  }

  public backward(): void {
    this.gradients.set(this, 0);
  }
}
