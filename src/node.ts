import * as bm from "bluemath";

/**
 * Base class for node in the network
 *
 * @export
 * @class Node
 */
export abstract class Node {
  public value: number | bm.NDArray | bm.Complex;

  public inboundNodes: Node[];
  public outboundNodes: Node[];
  // Keys are inputs to this node and their valies are partials of this node with respect to that input
  public gradients: Map<Node, number | bm.NDArray | bm.Complex> = new Map();

  /**
   * Creates an instance of Node.
   * @param {Node[]} [inboundNodes=[]]
   * @memberof Node
   */
  constructor(inboundNodes: Node[] = []) {
    // A list of nodes with edges into this nodes
    this.inboundNodes = inboundNodes;

    // A list of nodes that this node outputs to.
    this.outboundNodes = [];

    for (const n of inboundNodes) {
      n.outboundNodes.push(this);
    }
  }

  /**
   * Every node that uses this class as base class will need to define its own `forward` method
   *
   * @memberof Node
   */
  public forward(): void {
    throw new Error("Not Implemented");
  }

  /**
   * Every node that uses this class as base class will need to define its own `backward` method
   *
   * @memberof Node
   */
  public backward(): void {
    throw new Error("Not Implemented");
  }
}
